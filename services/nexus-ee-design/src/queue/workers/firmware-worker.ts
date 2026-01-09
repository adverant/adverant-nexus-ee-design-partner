/**
 * EE Design Partner - Firmware Worker
 *
 * BullMQ worker for processing firmware generation jobs.
 * Integrates with LLM for AI-assisted code generation.
 */

import { Worker, Job, ConnectionOptions } from 'bullmq';
import * as fs from 'fs/promises';
import * as path from 'path';
import axios from 'axios';
import { config } from '../../config.js';
import { log, Logger } from '../../utils/logger.js';
import * as FirmwareRepository from '../../database/repositories/firmware-repository.js';
import type { FirmwareJobData } from '../queue-manager.js';
import type {
  GeneratedFile,
  MCUFamily,
  RTOSConfig,
  HALConfig,
  BuildConfig,
  Driver,
  FirmwareTask,
} from '../../types/index.js';

// ============================================================================
// Types
// ============================================================================

interface FirmwareContext {
  logger: Logger;
  workDir: string;
  outputDir: string;
}

interface FirmwareResult {
  success: boolean;
  firmwareId: string;
  generatedFiles: GeneratedFile[];
  buildInstructions: string;
  warnings: string[];
  errors: string[];
  generationTime: number;
}

interface LLMResponse {
  content: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
  };
}

// ============================================================================
// Configuration
// ============================================================================

const getRedisConnection = (): ConnectionOptions => ({
  host: config.redis.host,
  port: config.redis.port,
  password: config.redis.password,
  maxRetriesPerRequest: null,
  enableReadyCheck: false,
});

const WORK_DIR = config.storage.tempDir;
const OUTPUT_DIR = config.storage.outputDir;
const OPENROUTER_API_KEY = config.llm.openrouterApiKey;
const PRIMARY_MODEL = config.llm.primaryModel;
const FAST_MODEL = config.llm.fastModel;

// ============================================================================
// Firmware Job Processor
// ============================================================================

async function processFirmwareJob(
  job: Job<FirmwareJobData>,
  context: FirmwareContext
): Promise<FirmwareResult> {
  const { logger } = context;
  const { firmwareId, name, targetMcu, rtosConfig, peripherals, buildConfig } = job.data;
  const startTime = Date.now();

  logger.info('Processing firmware generation job', {
    jobId: job.id,
    firmwareId,
    name,
    mcuFamily: targetMcu.family,
    mcuPart: targetMcu.part,
  });

  // Update status to generating
  await FirmwareRepository.update(firmwareId, { status: 'generating' });
  await job.updateProgress({ progress: 5, message: 'Starting firmware generation' });

  const generatedFiles: GeneratedFile[] = [];
  const warnings: string[] = [];
  const errors: string[] = [];

  try {
    // Step 1: Generate project structure
    await job.updateProgress({ progress: 10, message: 'Creating project structure' });
    const structureFiles = await generateProjectStructure(job.data, context);
    generatedFiles.push(...structureFiles);

    // Step 2: Generate HAL configuration
    await job.updateProgress({ progress: 25, message: 'Generating HAL configuration' });
    const halFiles = await generateHALCode(job.data, context);
    generatedFiles.push(...halFiles);

    // Step 3: Generate peripheral drivers (AI-assisted)
    await job.updateProgress({ progress: 40, message: 'Generating peripheral drivers' });
    const driverFiles = await generateDrivers(job.data, context);
    generatedFiles.push(...driverFiles);

    // Step 4: Generate RTOS configuration if enabled
    if (rtosConfig) {
      await job.updateProgress({ progress: 55, message: 'Configuring RTOS' });
      const rtosFiles = await generateRTOSConfig(job.data, context);
      generatedFiles.push(...rtosFiles);
    }

    // Step 5: Generate main application code
    await job.updateProgress({ progress: 70, message: 'Generating application code' });
    const appFiles = await generateApplicationCode(job.data, context);
    generatedFiles.push(...appFiles);

    // Step 6: Generate build configuration
    await job.updateProgress({ progress: 85, message: 'Creating build configuration' });
    const buildFiles = await generateBuildConfig(job.data, context);
    generatedFiles.push(...buildFiles);

    // Step 7: Update firmware repository with generated files
    await job.updateProgress({ progress: 95, message: 'Saving generated files' });
    await FirmwareRepository.updateSourceFiles(firmwareId, generatedFiles);

    // Generate build instructions
    const buildInstructions = generateBuildInstructions(job.data);

    // Update final status
    await FirmwareRepository.update(firmwareId, { status: 'generated' });
    await job.updateProgress({ progress: 100, message: 'Firmware generation complete' });

    const generationTime = Date.now() - startTime;

    logger.info('Firmware generation completed', {
      jobId: job.id,
      firmwareId,
      fileCount: generatedFiles.length,
      generationTime,
    });

    return {
      success: true,
      firmwareId,
      generatedFiles,
      buildInstructions,
      warnings,
      errors,
      generationTime,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    errors.push(errorMessage);

    // Update firmware status to reflect failure
    await FirmwareRepository.update(firmwareId, { status: 'draft' });

    const err = error instanceof Error ? error : new Error(errorMessage);
    logger.error('Firmware generation failed', err, {
      jobId: job.id,
      firmwareId,
    });

    throw error;
  }
}

// ============================================================================
// Project Structure Generation
// ============================================================================

async function generateProjectStructure(
  data: FirmwareJobData,
  context: FirmwareContext
): Promise<GeneratedFile[]> {
  const { name, targetMcu, rtosConfig } = data;
  const projectName = name.replace(/\s+/g, '_').toLowerCase();
  const files: GeneratedFile[] = [];

  // README.md
  files.push({
    path: 'README.md',
    type: 'config',
    content: `# ${name}

## Overview
Auto-generated firmware project for ${targetMcu.family.toUpperCase()} (${targetMcu.part})

## Target MCU
- Family: ${targetMcu.family}
- Part: ${targetMcu.part}
- Core: ${targetMcu.core}
- Flash: ${targetMcu.flashSize / 1024}KB
- RAM: ${targetMcu.ramSize / 1024}KB
- Clock: ${targetMcu.clockSpeed / 1e6}MHz

${rtosConfig ? `## RTOS
- Type: ${rtosConfig.type}
- Version: ${rtosConfig.version}
- Tick Rate: ${rtosConfig.tickRate}Hz
- Heap Size: ${rtosConfig.heapSize / 1024}KB
` : ''}

## Build Instructions
\`\`\`bash
mkdir build && cd build
cmake ..
make -j$(nproc)
\`\`\`

## Generated by
Nexus EE Design Partner - Firmware Generator
Generated: ${new Date().toISOString()}
`,
    generatedAt: new Date().toISOString(),
  });

  // .clang-format
  files.push({
    path: '.clang-format',
    type: 'config',
    content: `BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
BreakBeforeBraces: Linux
AllowShortFunctionsOnASingleLine: None
`,
    generatedAt: new Date().toISOString(),
  });

  // .gitignore
  files.push({
    path: '.gitignore',
    type: 'config',
    content: `# Build
build/
*.o
*.elf
*.hex
*.bin
*.map

# IDE
.vscode/
.idea/
*.swp
*~

# OS
.DS_Store
Thumbs.db
`,
    generatedAt: new Date().toISOString(),
  });

  return files;
}

// ============================================================================
// HAL Code Generation
// ============================================================================

async function generateHALCode(
  data: FirmwareJobData,
  context: FirmwareContext
): Promise<GeneratedFile[]> {
  const { targetMcu, peripherals } = data;
  const files: GeneratedFile[] = [];

  // Get MCU-specific includes based on family
  const halIncludes = getHALIncludes(targetMcu.family);

  // Generate main header
  files.push({
    path: 'Core/Inc/main.h',
    type: 'header',
    content: `/**
 * @file main.h
 * @brief Main application header
 * @generated Nexus EE Design Partner
 */

#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes */
${halIncludes.map(inc => `#include "${inc}"`).join('\n')}

/* Exported types */
/* Exported constants */
/* Exported macro */
/* Exported functions */
void Error_Handler(void);

/* Private defines */
${generatePinDefines(peripherals)}

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
`,
    generatedAt: new Date().toISOString(),
  });

  // Generate peripheral initialization headers/sources
  for (const peripheral of peripherals) {
    const initFile = await generatePeripheralInit(peripheral, targetMcu.family, context);
    files.push(initFile);
  }

  return files;
}

function getHALIncludes(family: string): string[] {
  const includes: Record<string, string[]> = {
    stm32: ['stm32h7xx_hal.h', 'stm32h7xx_hal_conf.h'],
    esp32: ['esp_system.h', 'esp_log.h', 'driver/gpio.h'],
    ti_tms320: ['driverlib.h', 'device.h'],
    infineon_aurix: ['Ifx_Types.h', 'IfxCpu.h'],
    nordic_nrf: ['nrf.h', 'nrfx.h'],
    rpi_pico: ['pico/stdlib.h', 'hardware/gpio.h'],
    nxp_imxrt: ['fsl_common.h', 'pin_mux.h', 'clock_config.h'],
  };

  return includes[family] || ['main.h'];
}

function generatePinDefines(peripherals: FirmwareJobData['peripherals']): string {
  const defines: string[] = [];

  for (const peripheral of peripherals) {
    const pins = peripheral.config.pins as Array<{ name: string; pin: string }> | undefined;
    if (pins) {
      for (const pin of pins) {
        defines.push(`#define ${peripheral.instance}_${pin.name}_Pin ${pin.pin}`);
      }
    }
  }

  return defines.join('\n') || '/* No pin definitions */';
}

async function generatePeripheralInit(
  peripheral: FirmwareJobData['peripherals'][0],
  family: string,
  context: FirmwareContext
): Promise<GeneratedFile> {
  const { logger } = context;
  const instanceNum = peripheral.instance.replace(/\D/g, '');
  const typeName = peripheral.type.toUpperCase();

  // Use LLM to generate peripheral initialization code
  let content: string;

  try {
    content = await generateCodeWithLLM(
      `Generate ${family.toUpperCase()} HAL initialization code for ${peripheral.type} peripheral instance ${peripheral.instance}.
Include proper error handling and configuration based on: ${JSON.stringify(peripheral.config)}
Return only the C code without explanations.`,
      context.logger
    );
  } catch {
    // Fallback to template-based generation
    content = generatePeripheralInitTemplate(peripheral, family);
  }

  return {
    path: `Core/Src/${peripheral.type}_${peripheral.instance.toLowerCase()}.c`,
    type: 'source',
    content: `/**
 * @file ${peripheral.type}_${peripheral.instance.toLowerCase()}.c
 * @brief ${typeName} ${peripheral.instance} initialization
 * @generated Nexus EE Design Partner
 */

#include "main.h"

${content}
`,
    generatedAt: new Date().toISOString(),
  };
}

function generatePeripheralInitTemplate(
  peripheral: FirmwareJobData['peripherals'][0],
  family: string
): string {
  const instanceNum = peripheral.instance.replace(/\D/g, '');
  const typeName = peripheral.type.toUpperCase();

  switch (peripheral.type) {
    case 'uart':
      return `
UART_HandleTypeDef huart${instanceNum};

void MX_USART${instanceNum}_UART_Init(void) {
  huart${instanceNum}.Instance = USART${instanceNum};
  huart${instanceNum}.Init.BaudRate = ${peripheral.config.baudRate || 115200};
  huart${instanceNum}.Init.WordLength = UART_WORDLENGTH_8B;
  huart${instanceNum}.Init.StopBits = UART_STOPBITS_1;
  huart${instanceNum}.Init.Parity = UART_PARITY_NONE;
  huart${instanceNum}.Init.Mode = UART_MODE_TX_RX;
  huart${instanceNum}.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart${instanceNum}.Init.OverSampling = UART_OVERSAMPLING_16;

  if (HAL_UART_Init(&huart${instanceNum}) != HAL_OK) {
    Error_Handler();
  }
}
`;

    case 'spi':
      return `
SPI_HandleTypeDef hspi${instanceNum};

void MX_SPI${instanceNum}_Init(void) {
  hspi${instanceNum}.Instance = SPI${instanceNum};
  hspi${instanceNum}.Init.Mode = SPI_MODE_MASTER;
  hspi${instanceNum}.Init.Direction = SPI_DIRECTION_2LINES;
  hspi${instanceNum}.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi${instanceNum}.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi${instanceNum}.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi${instanceNum}.Init.NSS = SPI_NSS_SOFT;
  hspi${instanceNum}.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
  hspi${instanceNum}.Init.FirstBit = SPI_FIRSTBIT_MSB;

  if (HAL_SPI_Init(&hspi${instanceNum}) != HAL_OK) {
    Error_Handler();
  }
}
`;

    case 'i2c':
      return `
I2C_HandleTypeDef hi2c${instanceNum};

void MX_I2C${instanceNum}_Init(void) {
  hi2c${instanceNum}.Instance = I2C${instanceNum};
  hi2c${instanceNum}.Init.Timing = 0x10909CEC;
  hi2c${instanceNum}.Init.OwnAddress1 = 0;
  hi2c${instanceNum}.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c${instanceNum}.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c${instanceNum}.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c${instanceNum}.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;

  if (HAL_I2C_Init(&hi2c${instanceNum}) != HAL_OK) {
    Error_Handler();
  }
}
`;

    case 'adc':
      return `
ADC_HandleTypeDef hadc${instanceNum};

void MX_ADC${instanceNum}_Init(void) {
  hadc${instanceNum}.Instance = ADC${instanceNum};
  hadc${instanceNum}.Init.Resolution = ADC_RESOLUTION_12B;
  hadc${instanceNum}.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc${instanceNum}.Init.ContinuousConvMode = DISABLE;
  hadc${instanceNum}.Init.ExternalTrigConv = ADC_SOFTWARE_START;

  if (HAL_ADC_Init(&hadc${instanceNum}) != HAL_OK) {
    Error_Handler();
  }
}
`;

    case 'pwm':
      return `
TIM_HandleTypeDef htim${instanceNum};

void MX_TIM${instanceNum}_PWM_Init(void) {
  htim${instanceNum}.Instance = TIM${instanceNum};
  htim${instanceNum}.Init.Prescaler = 0;
  htim${instanceNum}.Init.Period = 999;
  htim${instanceNum}.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim${instanceNum}.Init.RepetitionCounter = 0;

  if (HAL_TIM_PWM_Init(&htim${instanceNum}) != HAL_OK) {
    Error_Handler();
  }

  TIM_OC_InitTypeDef sConfigOC = {0};
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

  if (HAL_TIM_PWM_ConfigChannel(&htim${instanceNum}, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) {
    Error_Handler();
  }
}
`;

    default:
      return `
/* ${typeName} ${peripheral.instance} initialization */
void MX_${typeName}${instanceNum}_Init(void) {
  /* TODO: Add initialization code */
}
`;
  }
}

// ============================================================================
// Driver Generation (AI-Assisted)
// ============================================================================

async function generateDrivers(
  data: FirmwareJobData,
  context: FirmwareContext
): Promise<GeneratedFile[]> {
  const { peripherals } = data;
  const files: GeneratedFile[] = [];

  for (const peripheral of peripherals) {
    if (peripheral.connectedTo) {
      // Generate driver header
      const headerFile = await generateDriverHeader(peripheral, context);
      files.push(headerFile);

      // Generate driver source
      const sourceFile = await generateDriverSource(peripheral, context);
      files.push(sourceFile);
    }
  }

  return files;
}

async function generateDriverHeader(
  peripheral: FirmwareJobData['peripherals'][0],
  context: FirmwareContext
): Promise<GeneratedFile> {
  const componentName = peripheral.connectedTo!.replace(/\d+$/, '');
  const guardName = `__${componentName.toUpperCase()}_DRIVER_H`;

  return {
    path: `Drivers/Inc/${peripheral.connectedTo!.toLowerCase()}_driver.h`,
    type: 'header',
    content: `/**
 * @file ${peripheral.connectedTo!.toLowerCase()}_driver.h
 * @brief Driver for ${peripheral.connectedTo}
 * @generated Nexus EE Design Partner
 */

#ifndef ${guardName}
#define ${guardName}

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include <stdint.h>
#include <stdbool.h>

/* Type definitions */
typedef struct {
  uint8_t initialized;
  void* handle;
} ${componentName}_Handle_t;

typedef enum {
  ${componentName.toUpperCase()}_OK = 0,
  ${componentName.toUpperCase()}_ERROR = -1,
  ${componentName.toUpperCase()}_BUSY = -2,
  ${componentName.toUpperCase()}_TIMEOUT = -3,
} ${componentName}_Status_t;

/* Function prototypes */
${componentName}_Status_t ${componentName}_Init(${componentName}_Handle_t* handle);
${componentName}_Status_t ${componentName}_DeInit(${componentName}_Handle_t* handle);
${componentName}_Status_t ${componentName}_Read(${componentName}_Handle_t* handle, uint8_t* data, uint16_t len);
${componentName}_Status_t ${componentName}_Write(${componentName}_Handle_t* handle, const uint8_t* data, uint16_t len);
bool ${componentName}_IsReady(${componentName}_Handle_t* handle);

#ifdef __cplusplus
}
#endif

#endif /* ${guardName} */
`,
    generatedAt: new Date().toISOString(),
  };
}

async function generateDriverSource(
  peripheral: FirmwareJobData['peripherals'][0],
  context: FirmwareContext
): Promise<GeneratedFile> {
  const componentName = peripheral.connectedTo!.replace(/\d+$/, '');
  const { logger } = context;

  // Try to use LLM for more sophisticated driver generation
  let implementation: string;

  try {
    implementation = await generateCodeWithLLM(
      `Generate a complete C driver implementation for ${peripheral.connectedTo} connected via ${peripheral.type}.
The driver should include Init, DeInit, Read, Write, and IsReady functions.
Use proper error handling and return status codes.
Include any necessary register access or protocol handling for ${peripheral.type}.
Return only the function implementations without includes or type definitions.`,
      logger
    );
  } catch {
    // Fallback to template
    implementation = generateDriverTemplate(componentName);
  }

  return {
    path: `Drivers/Src/${peripheral.connectedTo!.toLowerCase()}_driver.c`,
    type: 'source',
    content: `/**
 * @file ${peripheral.connectedTo!.toLowerCase()}_driver.c
 * @brief Driver implementation for ${peripheral.connectedTo}
 * @generated Nexus EE Design Partner
 */

#include "${peripheral.connectedTo!.toLowerCase()}_driver.h"

${implementation}
`,
    generatedAt: new Date().toISOString(),
  };
}

function generateDriverTemplate(componentName: string): string {
  return `
${componentName}_Status_t ${componentName}_Init(${componentName}_Handle_t* handle) {
  if (handle == NULL) {
    return ${componentName.toUpperCase()}_ERROR;
  }

  /* Initialize hardware */
  /* TODO: Add initialization code */

  handle->initialized = 1;
  return ${componentName.toUpperCase()}_OK;
}

${componentName}_Status_t ${componentName}_DeInit(${componentName}_Handle_t* handle) {
  if (handle == NULL || !handle->initialized) {
    return ${componentName.toUpperCase()}_ERROR;
  }

  /* Deinitialize hardware */
  /* TODO: Add deinitialization code */

  handle->initialized = 0;
  return ${componentName.toUpperCase()}_OK;
}

${componentName}_Status_t ${componentName}_Read(${componentName}_Handle_t* handle, uint8_t* data, uint16_t len) {
  if (handle == NULL || !handle->initialized || data == NULL) {
    return ${componentName.toUpperCase()}_ERROR;
  }

  /* Read data from device */
  /* TODO: Add read implementation */

  return ${componentName.toUpperCase()}_OK;
}

${componentName}_Status_t ${componentName}_Write(${componentName}_Handle_t* handle, const uint8_t* data, uint16_t len) {
  if (handle == NULL || !handle->initialized || data == NULL) {
    return ${componentName.toUpperCase()}_ERROR;
  }

  /* Write data to device */
  /* TODO: Add write implementation */

  return ${componentName.toUpperCase()}_OK;
}

bool ${componentName}_IsReady(${componentName}_Handle_t* handle) {
  if (handle == NULL) {
    return false;
  }
  return handle->initialized != 0;
}
`;
}

// ============================================================================
// RTOS Configuration Generation
// ============================================================================

async function generateRTOSConfig(
  data: FirmwareJobData,
  context: FirmwareContext
): Promise<GeneratedFile[]> {
  const { targetMcu, rtosConfig, peripherals } = data;
  const files: GeneratedFile[] = [];

  if (!rtosConfig) return files;

  // Generate FreeRTOS config (most common)
  if (rtosConfig.type === 'freertos') {
    files.push({
      path: 'Core/Inc/FreeRTOSConfig.h',
      type: 'header',
      content: generateFreeRTOSConfig(targetMcu, rtosConfig),
      generatedAt: new Date().toISOString(),
    });
  }

  // Generate tasks file
  const tasks = generateTaskList(peripherals);
  files.push({
    path: 'Core/Src/app_tasks.c',
    type: 'source',
    content: generateTasksCode(rtosConfig, tasks),
    generatedAt: new Date().toISOString(),
  });

  files.push({
    path: 'Core/Inc/app_tasks.h',
    type: 'header',
    content: generateTasksHeader(tasks),
    generatedAt: new Date().toISOString(),
  });

  return files;
}

function generateFreeRTOSConfig(
  targetMcu: FirmwareJobData['targetMcu'],
  rtosConfig: NonNullable<FirmwareJobData['rtosConfig']>
): string {
  return `/**
 * @file FreeRTOSConfig.h
 * @brief FreeRTOS configuration
 * @generated Nexus EE Design Partner
 */

#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H

/* Basic FreeRTOS configuration */
#define configUSE_PREEMPTION                    1
#define configUSE_PORT_OPTIMISED_TASK_SELECTION 0
#define configUSE_TICKLESS_IDLE                 0
#define configCPU_CLOCK_HZ                      ${targetMcu.clockSpeed}UL
#define configTICK_RATE_HZ                      ${rtosConfig.tickRate}
#define configMAX_PRIORITIES                    ${Math.min(rtosConfig.maxTasks, 7)}
#define configMINIMAL_STACK_SIZE                128
#define configTOTAL_HEAP_SIZE                   ${rtosConfig.heapSize}
#define configMAX_TASK_NAME_LEN                 16
#define configUSE_16_BIT_TICKS                  0
#define configIDLE_SHOULD_YIELD                 1
#define configUSE_TASK_NOTIFICATIONS            1
#define configUSE_MUTEXES                       1
#define configUSE_RECURSIVE_MUTEXES             1
#define configUSE_COUNTING_SEMAPHORES           1
#define configQUEUE_REGISTRY_SIZE               8
#define configUSE_QUEUE_SETS                    0
#define configUSE_TIME_SLICING                  1
#define configUSE_NEWLIB_REENTRANT              0
#define configENABLE_BACKWARD_COMPATIBILITY     0
#define configNUM_THREAD_LOCAL_STORAGE_POINTERS 5

/* Hook function related definitions */
#define configUSE_IDLE_HOOK                     0
#define configUSE_TICK_HOOK                     0
#define configCHECK_FOR_STACK_OVERFLOW          2
#define configUSE_MALLOC_FAILED_HOOK            1
#define configUSE_DAEMON_TASK_STARTUP_HOOK      0

/* Memory allocation related definitions */
#define configSUPPORT_STATIC_ALLOCATION         1
#define configSUPPORT_DYNAMIC_ALLOCATION        1

/* Interrupt nesting behaviour configuration */
#define configKERNEL_INTERRUPT_PRIORITY         255
#define configMAX_SYSCALL_INTERRUPT_PRIORITY    191
#define configMAX_API_CALL_INTERRUPT_PRIORITY   191

/* Software timer definitions */
#define configUSE_TIMERS                        1
#define configTIMER_TASK_PRIORITY               2
#define configTIMER_QUEUE_LENGTH                10
#define configTIMER_TASK_STACK_DEPTH            256

/* Define to trap errors during development */
#define configASSERT( x ) if( ( x ) == 0 ) { taskDISABLE_INTERRUPTS(); for( ;; ); }

/* Optional functions */
#define INCLUDE_vTaskPrioritySet                1
#define INCLUDE_uxTaskPriorityGet               1
#define INCLUDE_vTaskDelete                     1
#define INCLUDE_vTaskSuspend                    1
#define INCLUDE_vTaskDelayUntil                 1
#define INCLUDE_vTaskDelay                      1

#endif /* FREERTOS_CONFIG_H */
`;
}

interface TaskDef {
  name: string;
  priority: number;
  stackSize: number;
  period: number;
  description: string;
}

function generateTaskList(peripherals: FirmwareJobData['peripherals']): TaskDef[] {
  const tasks: TaskDef[] = [];

  // Main task
  tasks.push({
    name: 'MainTask',
    priority: 2,
    stackSize: 512,
    period: 100,
    description: 'Main application task',
  });

  // Communication task if UART/SPI/I2C/CAN present
  const hasComm = peripherals.some((p) =>
    ['uart', 'spi', 'i2c', 'can'].includes(p.type)
  );
  if (hasComm) {
    tasks.push({
      name: 'CommTask',
      priority: 3,
      stackSize: 512,
      period: 10,
      description: 'Communication handling task',
    });
  }

  // Sensor task if ADC present
  const hasADC = peripherals.some((p) => p.type === 'adc');
  if (hasADC) {
    tasks.push({
      name: 'SensorTask',
      priority: 2,
      stackSize: 256,
      period: 50,
      description: 'Sensor reading task',
    });
  }

  return tasks;
}

function generateTasksCode(
  rtosConfig: NonNullable<FirmwareJobData['rtosConfig']>,
  tasks: TaskDef[]
): string {
  const taskFunctions = tasks
    .map(
      (task) => `
/**
 * @brief ${task.description}
 */
void ${task.name}_Entry(void *pvParameters) {
  (void)pvParameters;

  /* Task initialization */

  for (;;) {
    /* Task body */

    vTaskDelay(pdMS_TO_TICKS(${task.period}));
  }
}
`
    )
    .join('\n');

  const taskCreation = tasks
    .map(
      (task) =>
        `  xTaskCreate(${task.name}_Entry, "${task.name}", ${task.stackSize}, NULL, ${task.priority}, NULL);`
    )
    .join('\n');

  return `/**
 * @file app_tasks.c
 * @brief RTOS task implementations
 * @generated Nexus EE Design Partner
 */

#include "app_tasks.h"
#include "FreeRTOS.h"
#include "task.h"

/* Task functions */
${taskFunctions}

/**
 * @brief Create all application tasks
 */
void App_CreateTasks(void) {
${taskCreation}
}
`;
}

function generateTasksHeader(tasks: TaskDef[]): string {
  const prototypes = tasks
    .map((task) => `void ${task.name}_Entry(void *pvParameters);`)
    .join('\n');

  return `/**
 * @file app_tasks.h
 * @brief RTOS task declarations
 * @generated Nexus EE Design Partner
 */

#ifndef __APP_TASKS_H
#define __APP_TASKS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Task entry point prototypes */
${prototypes}

/* Task creation function */
void App_CreateTasks(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_TASKS_H */
`;
}

// ============================================================================
// Application Code Generation
// ============================================================================

async function generateApplicationCode(
  data: FirmwareJobData,
  context: FirmwareContext
): Promise<GeneratedFile[]> {
  const { targetMcu, rtosConfig, peripherals } = data;
  const files: GeneratedFile[] = [];

  // Generate peripheral init calls
  const peripheralInit = peripherals
    .map((p) => {
      const instanceNum = p.instance.replace(/\D/g, '');
      return `  MX_${p.type.toUpperCase()}${instanceNum}_Init();`;
    })
    .join('\n');

  // Generate RTOS start code if enabled
  const rtosStart = rtosConfig
    ? `  App_CreateTasks();
  vTaskStartScheduler();`
    : '';

  // Generate main loop code
  const mainLoop = rtosConfig ? '/* RTOS running */' : '/* Main loop */';

  files.push({
    path: 'Core/Src/main.c',
    type: 'source',
    content: `/**
 * @file main.c
 * @brief Main application entry point
 * @generated Nexus EE Design Partner
 */

/* Includes */
#include "main.h"
${rtosConfig ? '#include "FreeRTOS.h"\n#include "task.h"\n#include "app_tasks.h"' : ''}

/* Private function prototypes */
void SystemClock_Config(void);
void Error_Handler(void);
${peripherals.map((p) => {
  const instanceNum = p.instance.replace(/\D/g, '');
  return `void MX_${p.type.toUpperCase()}${instanceNum}_Init(void);`;
}).join('\n')}

/**
 * @brief Application entry point
 */
int main(void) {
  /* MCU Configuration */
  HAL_Init();
  SystemClock_Config();

  /* Initialize all configured peripherals */
${peripheralInit}

  /* Start RTOS scheduler if enabled */
${rtosStart}

  /* Infinite loop */
  while (1) {
    ${mainLoop}
  }
}

/**
 * @brief System Clock Configuration
 */
void SystemClock_Config(void) {
  /* TODO: Configure system clocks */
}

/**
 * @brief Error Handler
 */
void Error_Handler(void) {
  __disable_irq();
  while (1) {
    /* Stay here */
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {
  /* User can add implementation */
}
#endif
`,
    generatedAt: new Date().toISOString(),
  });

  return files;
}

// ============================================================================
// Build Configuration Generation
// ============================================================================

async function generateBuildConfig(
  data: FirmwareJobData,
  context: FirmwareContext
): Promise<GeneratedFile[]> {
  const { name, targetMcu, rtosConfig, buildConfig } = data;
  const projectName = name.replace(/\s+/g, '_').toLowerCase();
  const files: GeneratedFile[] = [];

  const effectiveBuildConfig = buildConfig || {
    toolchain: 'gcc-arm',
    buildSystem: 'cmake',
    optimizationLevel: 'O2',
    debugSymbols: true,
    defines: { USE_HAL_DRIVER: '1' },
  };

  // CMakeLists.txt
  files.push({
    path: 'CMakeLists.txt',
    type: 'build',
    content: `# CMakeLists.txt
# Generated by Nexus EE Design Partner

cmake_minimum_required(VERSION 3.20)

# Project name
project(${projectName} C ASM)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# MCU specific flags
set(MCU_FLAGS "-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard")

# Compiler flags
set(CMAKE_C_FLAGS "\${CMAKE_C_FLAGS} \${MCU_FLAGS}")
set(CMAKE_C_FLAGS "\${CMAKE_C_FLAGS} -Wall -fdata-sections -ffunction-sections")
set(CMAKE_C_FLAGS_DEBUG "-g -${effectiveBuildConfig.debugSymbols ? 'Og' : 'O0'}")
set(CMAKE_C_FLAGS_RELEASE "-${effectiveBuildConfig.optimizationLevel}")

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS "\${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -specs=nosys.specs")

# Include directories
include_directories(
  Core/Inc
  Drivers/Inc
  ${rtosConfig ? 'Middlewares/Third_Party/FreeRTOS/Source/include\n  Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1' : ''}
)

# Source files
file(GLOB_RECURSE SOURCES
  "Core/Src/*.c"
  "Drivers/Src/*.c"
  ${rtosConfig ? '"Middlewares/Third_Party/FreeRTOS/Source/*.c"\n  "Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1/*.c"' : ''}
)

# Startup file
set(STARTUP_FILE startup_${targetMcu.part.toLowerCase()}.s)

# Linker script
set(LINKER_SCRIPT ${targetMcu.part.toUpperCase()}_FLASH.ld)

# Define executable
add_executable(\${PROJECT_NAME}.elf \${SOURCES} \${STARTUP_FILE})

# Link options
target_link_options(\${PROJECT_NAME}.elf PRIVATE
  -T\${CMAKE_SOURCE_DIR}/\${LINKER_SCRIPT}
)

# Generate hex and bin files
add_custom_command(TARGET \${PROJECT_NAME}.elf POST_BUILD
  COMMAND \${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:\${PROJECT_NAME}.elf> \${PROJECT_NAME}.hex
  COMMAND \${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:\${PROJECT_NAME}.elf> \${PROJECT_NAME}.bin
  COMMAND \${CMAKE_SIZE} $<TARGET_FILE:\${PROJECT_NAME}.elf>
)
`,
    generatedAt: new Date().toISOString(),
  });

  // Makefile
  files.push({
    path: 'Makefile',
    type: 'build',
    content: `# Makefile
# Generated by Nexus EE Design Partner

TARGET = ${projectName}

BUILD_DIR = build
C_SOURCES = $(wildcard Core/Src/*.c) $(wildcard Drivers/Src/*.c)
C_INCLUDES = -ICore/Inc -IDrivers/Inc

PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
AS = $(PREFIX)gcc -x assembler-with-cpp
OBJCOPY = $(PREFIX)objcopy
SIZE = $(PREFIX)size

CPU = -mcpu=cortex-m7
FPU = -mfpu=fpv5-d16
FLOAT-ABI = -mfloat-abi=hard
MCU = $(CPU) -mthumb $(FPU) $(FLOAT-ABI)

C_DEFS = ${Object.entries(effectiveBuildConfig.defines || {}).map(([k, v]) => `-D${k}=${v}`).join(' ')}

OPT = -${effectiveBuildConfig.optimizationLevel}

CFLAGS = $(MCU) $(C_DEFS) $(C_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections
${effectiveBuildConfig.debugSymbols ? 'CFLAGS += -g -gdwarf-2' : ''}

all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).hex $(BUILD_DIR)/$(TARGET).bin

OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))
vpath %.c $(sort $(dir $(C_SOURCES)))

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
\t$(CC) -c $(CFLAGS) $< -o $@

$(BUILD_DIR)/$(TARGET).elf: $(OBJECTS)
\t$(CC) $(OBJECTS) $(MCU) -specs=nosys.specs -Wl,--gc-sections -o $@
\t$(SIZE) $@

$(BUILD_DIR)/%.hex: $(BUILD_DIR)/%.elf
\t$(OBJCOPY) -O ihex $< $@

$(BUILD_DIR)/%.bin: $(BUILD_DIR)/%.elf
\t$(OBJCOPY) -O binary $< $@

$(BUILD_DIR):
\tmkdir -p $@

clean:
\t-rm -rf $(BUILD_DIR)

.PHONY: all clean
`,
    generatedAt: new Date().toISOString(),
  });

  return files;
}

function generateBuildInstructions(data: FirmwareJobData): string {
  const projectName = data.name.replace(/\s+/g, '_').toLowerCase();

  return `# Build Instructions for ${data.name}

## Prerequisites
- ARM GCC toolchain (arm-none-eabi-gcc)
- CMake 3.20+
- Make or Ninja

## Building with CMake
\`\`\`bash
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/arm-none-eabi.cmake ..
make -j$(nproc)
\`\`\`

## Building with Make
\`\`\`bash
make
\`\`\`

## Flashing
\`\`\`bash
# Using OpenOCD
openocd -f interface/stlink.cfg -f target/${data.targetMcu.family}.cfg -c "program build/${projectName}.elf verify reset exit"

# Using ST-Link
st-flash write build/${projectName}.bin 0x08000000
\`\`\`
`;
}

// ============================================================================
// LLM Integration
// ============================================================================

async function generateCodeWithLLM(
  prompt: string,
  logger: Logger
): Promise<string> {
  if (!OPENROUTER_API_KEY) {
    throw new Error('OpenRouter API key not configured');
  }

  try {
    const response = await axios.post<{
      choices: Array<{ message: { content: string } }>;
      usage?: { prompt_tokens: number; completion_tokens: number };
    }>(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model: FAST_MODEL,
        messages: [
          {
            role: 'system',
            content: 'You are an expert embedded systems firmware developer. Generate clean, production-ready C code following best practices. Return only code without explanations.',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
        max_tokens: 2000,
        temperature: 0.3,
      },
      {
        headers: {
          'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://adverant.ai',
          'X-Title': 'Nexus EE Design Partner',
        },
        timeout: 30000,
      }
    );

    const content = response.data.choices[0]?.message?.content || '';

    logger.debug('LLM code generation completed', {
      promptLength: prompt.length,
      responseLength: content.length,
      usage: response.data.usage,
    });

    return content;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.warn('LLM code generation failed, using template', { error: errorMessage });
    throw error;
  }
}

// ============================================================================
// Worker Creation
// ============================================================================

let firmwareWorker: Worker | null = null;

export function createFirmwareWorker(): Worker {
  const logger = log.child({ service: 'firmware-worker' });

  logger.info('Creating firmware worker', {
    concurrency: 2,
  });

  firmwareWorker = new Worker<FirmwareJobData, FirmwareResult>(
    'firmware',
    async (job) => {
      const context: FirmwareContext = {
        logger: logger.child({ jobId: job.id }),
        workDir: WORK_DIR,
        outputDir: OUTPUT_DIR,
      };

      return processFirmwareJob(job, context);
    },
    {
      connection: getRedisConnection(),
      concurrency: 2,
      limiter: {
        max: 4,
        duration: 1000,
      },
    }
  );

  // Set up worker event handlers
  firmwareWorker.on('completed', (job, result) => {
    logger.info('Firmware job completed', {
      jobId: job.id,
      firmwareId: job.data.firmwareId,
      fileCount: result.generatedFiles.length,
      generationTime: result.generationTime,
    });
  });

  firmwareWorker.on('failed', (job, error) => {
    logger.error('Firmware job failed', error, {
      jobId: job?.id,
      firmwareId: job?.data.firmwareId,
      attemptsMade: job?.attemptsMade,
    });
  });

  firmwareWorker.on('stalled', (jobId) => {
    logger.warn('Firmware job stalled', { jobId });
  });

  firmwareWorker.on('error', (error) => {
    logger.error('Firmware worker error', error);
  });

  return firmwareWorker;
}

export function getFirmwareWorker(): Worker | null {
  return firmwareWorker;
}

export async function closeFirmwareWorker(): Promise<void> {
  if (firmwareWorker) {
    await firmwareWorker.close();
    firmwareWorker = null;
    log.info('Firmware worker closed');
  }
}

export default createFirmwareWorker;
