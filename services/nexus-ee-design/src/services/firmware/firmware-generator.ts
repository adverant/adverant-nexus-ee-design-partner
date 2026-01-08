/**
 * Firmware Generator Service
 *
 * AI-assisted firmware generation for embedded systems.
 * Generates HAL code, drivers, RTOS configuration, and build systems.
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../../utils/logger';
import { ServiceError, ErrorCodes } from '../../utils/errors';
import {
  FirmwareProject,
  MCUTarget,
  MCUFamily,
  RTOSConfig,
  HALConfig,
  PeripheralConfig,
  PeripheralType,
  PinMapping,
  Driver,
  DriverFunction,
  FirmwareTask,
  BuildConfig,
  GeneratedFile,
  Schematic,
  Component
} from '../../types';

// ============================================================================
// Types
// ============================================================================

export interface FirmwareGeneratorConfig {
  templatesPath: string;
  outputPath: string;
  enableAIAssist: boolean;
  supportedFamilies: MCUFamily[];
  defaultRTOS: 'freertos' | 'zephyr';
}

export interface FirmwareRequirements {
  projectName: string;
  targetMcu: MCUTarget;
  schematic?: Schematic;
  rtos?: RTOSConfig;
  peripherals: PeripheralRequirement[];
  features?: FeatureRequirement[];
  buildConfig?: Partial<BuildConfig>;
}

export interface PeripheralRequirement {
  type: PeripheralType;
  instance: string;
  config: Record<string, unknown>;
  connectedTo?: string; // Component reference from schematic
}

export interface FeatureRequirement {
  name: string;
  type: 'protocol' | 'algorithm' | 'driver' | 'middleware';
  parameters?: Record<string, unknown>;
}

export interface GenerationResult {
  success: boolean;
  firmwareProject?: FirmwareProject;
  generatedFiles: GeneratedFile[];
  buildInstructions: string;
  warnings: string[];
  errors: string[];
  generationTime: number;
}

export interface HALTemplate {
  family: MCUFamily;
  vendor: string;
  halName: string;
  version: string;
  includes: string[];
  initCode: string;
  peripheralTemplates: Record<PeripheralType, PeripheralTemplate>;
}

export interface PeripheralTemplate {
  structName: string;
  initFunction: string;
  configStruct: string;
  headerIncludes: string[];
  sourceIncludes: string[];
  initTemplate: string;
  functionTemplates: Record<string, string>;
}

export interface DriverTemplate {
  name: string;
  interface: PeripheralType;
  headerTemplate: string;
  sourceTemplate: string;
  functions: DriverFunctionTemplate[];
}

export interface DriverFunctionTemplate {
  name: string;
  signature: string;
  implementation: string;
}

// ============================================================================
// MCU Family Configurations
// ============================================================================

const MCU_CONFIGS: Record<MCUFamily, HALTemplate> = {
  stm32: {
    family: 'stm32',
    vendor: 'STMicroelectronics',
    halName: 'STM32 HAL',
    version: '1.0.0',
    includes: ['stm32h7xx_hal.h', 'stm32h7xx_hal_conf.h'],
    initCode: `
void SystemClock_Config(void);
void MX_GPIO_Init(void);

int main(void) {
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();

  /* Initialize all configured peripherals */
  {{PERIPHERAL_INIT}}

  /* Start RTOS scheduler if enabled */
  {{RTOS_START}}

  while (1) {
    {{MAIN_LOOP}}
  }
}
`,
    peripheralTemplates: {
      gpio: {
        structName: 'GPIO_InitTypeDef',
        initFunction: 'MX_GPIO_Init',
        configStruct: `
GPIO_InitTypeDef GPIO_InitStruct = {0};
GPIO_InitStruct.Pin = {{PIN}};
GPIO_InitStruct.Mode = {{MODE}};
GPIO_InitStruct.Pull = {{PULL}};
GPIO_InitStruct.Speed = {{SPEED}};`,
        headerIncludes: ['stm32h7xx_hal_gpio.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_GPIO_Init(void) {
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  {{GPIO_CONFIG}}
  HAL_GPIO_Init({{PORT}}, &GPIO_InitStruct);
}`,
        functionTemplates: {
          read: 'HAL_GPIO_ReadPin({{PORT}}, {{PIN}})',
          write: 'HAL_GPIO_WritePin({{PORT}}, {{PIN}}, {{STATE}})',
          toggle: 'HAL_GPIO_TogglePin({{PORT}}, {{PIN}})'
        }
      },
      uart: {
        structName: 'UART_HandleTypeDef',
        initFunction: 'MX_USART{{N}}_UART_Init',
        configStruct: `
UART_HandleTypeDef huart{{N}};
huart{{N}}.Instance = USART{{N}};
huart{{N}}.Init.BaudRate = {{BAUDRATE}};
huart{{N}}.Init.WordLength = {{WORDLENGTH}};
huart{{N}}.Init.StopBits = {{STOPBITS}};
huart{{N}}.Init.Parity = {{PARITY}};
huart{{N}}.Init.Mode = UART_MODE_TX_RX;
huart{{N}}.Init.HwFlowCtl = {{HWFLOWCTL}};`,
        headerIncludes: ['stm32h7xx_hal_uart.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_USART{{N}}_UART_Init(void) {
  {{UART_CONFIG}}
  if (HAL_UART_Init(&huart{{N}}) != HAL_OK) {
    Error_Handler();
  }
}`,
        functionTemplates: {
          transmit: 'HAL_UART_Transmit(&huart{{N}}, {{DATA}}, {{SIZE}}, {{TIMEOUT}})',
          receive: 'HAL_UART_Receive(&huart{{N}}, {{DATA}}, {{SIZE}}, {{TIMEOUT}})',
          transmitIT: 'HAL_UART_Transmit_IT(&huart{{N}}, {{DATA}}, {{SIZE}})',
          receiveIT: 'HAL_UART_Receive_IT(&huart{{N}}, {{DATA}}, {{SIZE}})'
        }
      },
      spi: {
        structName: 'SPI_HandleTypeDef',
        initFunction: 'MX_SPI{{N}}_Init',
        configStruct: `
SPI_HandleTypeDef hspi{{N}};
hspi{{N}}.Instance = SPI{{N}};
hspi{{N}}.Init.Mode = {{MODE}};
hspi{{N}}.Init.Direction = {{DIRECTION}};
hspi{{N}}.Init.DataSize = {{DATASIZE}};
hspi{{N}}.Init.CLKPolarity = {{CPOL}};
hspi{{N}}.Init.CLKPhase = {{CPHA}};
hspi{{N}}.Init.NSS = {{NSS}};
hspi{{N}}.Init.BaudRatePrescaler = {{PRESCALER}};`,
        headerIncludes: ['stm32h7xx_hal_spi.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_SPI{{N}}_Init(void) {
  {{SPI_CONFIG}}
  if (HAL_SPI_Init(&hspi{{N}}) != HAL_OK) {
    Error_Handler();
  }
}`,
        functionTemplates: {
          transmit: 'HAL_SPI_Transmit(&hspi{{N}}, {{DATA}}, {{SIZE}}, {{TIMEOUT}})',
          receive: 'HAL_SPI_Receive(&hspi{{N}}, {{DATA}}, {{SIZE}}, {{TIMEOUT}})',
          transmitReceive: 'HAL_SPI_TransmitReceive(&hspi{{N}}, {{TX_DATA}}, {{RX_DATA}}, {{SIZE}}, {{TIMEOUT}})'
        }
      },
      i2c: {
        structName: 'I2C_HandleTypeDef',
        initFunction: 'MX_I2C{{N}}_Init',
        configStruct: `
I2C_HandleTypeDef hi2c{{N}};
hi2c{{N}}.Instance = I2C{{N}};
hi2c{{N}}.Init.Timing = {{TIMING}};
hi2c{{N}}.Init.OwnAddress1 = {{OWNADDR}};
hi2c{{N}}.Init.AddressingMode = {{ADDRMODE}};`,
        headerIncludes: ['stm32h7xx_hal_i2c.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_I2C{{N}}_Init(void) {
  {{I2C_CONFIG}}
  if (HAL_I2C_Init(&hi2c{{N}}) != HAL_OK) {
    Error_Handler();
  }
}`,
        functionTemplates: {
          masterTransmit: 'HAL_I2C_Master_Transmit(&hi2c{{N}}, {{ADDR}}, {{DATA}}, {{SIZE}}, {{TIMEOUT}})',
          masterReceive: 'HAL_I2C_Master_Receive(&hi2c{{N}}, {{ADDR}}, {{DATA}}, {{SIZE}}, {{TIMEOUT}})',
          memWrite: 'HAL_I2C_Mem_Write(&hi2c{{N}}, {{ADDR}}, {{MEMADDR}}, {{MEMSIZE}}, {{DATA}}, {{SIZE}}, {{TIMEOUT}})',
          memRead: 'HAL_I2C_Mem_Read(&hi2c{{N}}, {{ADDR}}, {{MEMADDR}}, {{MEMSIZE}}, {{DATA}}, {{SIZE}}, {{TIMEOUT}})'
        }
      },
      adc: {
        structName: 'ADC_HandleTypeDef',
        initFunction: 'MX_ADC{{N}}_Init',
        configStruct: `
ADC_HandleTypeDef hadc{{N}};
hadc{{N}}.Instance = ADC{{N}};
hadc{{N}}.Init.Resolution = {{RESOLUTION}};
hadc{{N}}.Init.ScanConvMode = {{SCANMODE}};
hadc{{N}}.Init.ContinuousConvMode = {{CONTINUOUS}};
hadc{{N}}.Init.ExternalTrigConv = {{TRIGGER}};`,
        headerIncludes: ['stm32h7xx_hal_adc.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_ADC{{N}}_Init(void) {
  {{ADC_CONFIG}}
  if (HAL_ADC_Init(&hadc{{N}}) != HAL_OK) {
    Error_Handler();
  }
  {{CHANNEL_CONFIG}}
}`,
        functionTemplates: {
          start: 'HAL_ADC_Start(&hadc{{N}})',
          stop: 'HAL_ADC_Stop(&hadc{{N}})',
          getValue: 'HAL_ADC_GetValue(&hadc{{N}})',
          poll: 'HAL_ADC_PollForConversion(&hadc{{N}}, {{TIMEOUT}})'
        }
      },
      dac: {
        structName: 'DAC_HandleTypeDef',
        initFunction: 'MX_DAC{{N}}_Init',
        configStruct: `
DAC_HandleTypeDef hdac{{N}};
hdac{{N}}.Instance = DAC{{N}};`,
        headerIncludes: ['stm32h7xx_hal_dac.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_DAC{{N}}_Init(void) {
  {{DAC_CONFIG}}
  if (HAL_DAC_Init(&hdac{{N}}) != HAL_OK) {
    Error_Handler();
  }
}`,
        functionTemplates: {
          start: 'HAL_DAC_Start(&hdac{{N}}, {{CHANNEL}})',
          setValue: 'HAL_DAC_SetValue(&hdac{{N}}, {{CHANNEL}}, {{ALIGN}}, {{VALUE}})'
        }
      },
      pwm: {
        structName: 'TIM_HandleTypeDef',
        initFunction: 'MX_TIM{{N}}_Init',
        configStruct: `
TIM_HandleTypeDef htim{{N}};
htim{{N}}.Instance = TIM{{N}};
htim{{N}}.Init.Prescaler = {{PRESCALER}};
htim{{N}}.Init.Period = {{PERIOD}};`,
        headerIncludes: ['stm32h7xx_hal_tim.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_TIM{{N}}_Init(void) {
  {{TIM_CONFIG}}
  if (HAL_TIM_PWM_Init(&htim{{N}}) != HAL_OK) {
    Error_Handler();
  }
  {{CHANNEL_CONFIG}}
}`,
        functionTemplates: {
          start: 'HAL_TIM_PWM_Start(&htim{{N}}, {{CHANNEL}})',
          stop: 'HAL_TIM_PWM_Stop(&htim{{N}}, {{CHANNEL}})',
          setDuty: '__HAL_TIM_SET_COMPARE(&htim{{N}}, {{CHANNEL}}, {{VALUE}})'
        }
      },
      timer: {
        structName: 'TIM_HandleTypeDef',
        initFunction: 'MX_TIM{{N}}_Init',
        configStruct: `
TIM_HandleTypeDef htim{{N}};
htim{{N}}.Instance = TIM{{N}};
htim{{N}}.Init.Prescaler = {{PRESCALER}};
htim{{N}}.Init.Period = {{PERIOD}};`,
        headerIncludes: ['stm32h7xx_hal_tim.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_TIM{{N}}_Init(void) {
  {{TIM_CONFIG}}
  if (HAL_TIM_Base_Init(&htim{{N}}) != HAL_OK) {
    Error_Handler();
  }
}`,
        functionTemplates: {
          start: 'HAL_TIM_Base_Start(&htim{{N}})',
          startIT: 'HAL_TIM_Base_Start_IT(&htim{{N}})',
          stop: 'HAL_TIM_Base_Stop(&htim{{N}})'
        }
      },
      can: {
        structName: 'FDCAN_HandleTypeDef',
        initFunction: 'MX_FDCAN{{N}}_Init',
        configStruct: `
FDCAN_HandleTypeDef hfdcan{{N}};
hfdcan{{N}}.Instance = FDCAN{{N}};
hfdcan{{N}}.Init.NominalPrescaler = {{PRESCALER}};
hfdcan{{N}}.Init.NominalSyncJumpWidth = {{SJW}};
hfdcan{{N}}.Init.NominalTimeSeg1 = {{TSEG1}};
hfdcan{{N}}.Init.NominalTimeSeg2 = {{TSEG2}};`,
        headerIncludes: ['stm32h7xx_hal_fdcan.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_FDCAN{{N}}_Init(void) {
  {{FDCAN_CONFIG}}
  if (HAL_FDCAN_Init(&hfdcan{{N}}) != HAL_OK) {
    Error_Handler();
  }
  {{FILTER_CONFIG}}
}`,
        functionTemplates: {
          start: 'HAL_FDCAN_Start(&hfdcan{{N}})',
          transmit: 'HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan{{N}}, {{HEADER}}, {{DATA}})',
          receive: 'HAL_FDCAN_GetRxMessage(&hfdcan{{N}}, {{FIFO}}, {{HEADER}}, {{DATA}})'
        }
      },
      usb: {
        structName: 'PCD_HandleTypeDef',
        initFunction: 'MX_USB_OTG_FS_PCD_Init',
        configStruct: `
PCD_HandleTypeDef hpcd_USB_OTG_FS;
hpcd_USB_OTG_FS.Instance = USB_OTG_FS;`,
        headerIncludes: ['stm32h7xx_hal_pcd.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_USB_OTG_FS_PCD_Init(void) {
  {{USB_CONFIG}}
  if (HAL_PCD_Init(&hpcd_USB_OTG_FS) != HAL_OK) {
    Error_Handler();
  }
}`,
        functionTemplates: {}
      },
      ethernet: {
        structName: 'ETH_HandleTypeDef',
        initFunction: 'MX_ETH_Init',
        configStruct: `
ETH_HandleTypeDef heth;
heth.Instance = ETH;`,
        headerIncludes: ['stm32h7xx_hal_eth.h'],
        sourceIncludes: [],
        initTemplate: `
void MX_ETH_Init(void) {
  {{ETH_CONFIG}}
  if (HAL_ETH_Init(&heth) != HAL_OK) {
    Error_Handler();
  }
}`,
        functionTemplates: {}
      }
    }
  },
  esp32: {
    family: 'esp32',
    vendor: 'Espressif',
    halName: 'ESP-IDF',
    version: '5.0.0',
    includes: ['esp_system.h', 'esp_log.h', 'driver/gpio.h'],
    initCode: `
void app_main(void) {
  ESP_LOGI(TAG, "Starting application");

  /* Initialize NVS */
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  /* Initialize all configured peripherals */
  {{PERIPHERAL_INIT}}

  /* Start RTOS tasks if enabled */
  {{RTOS_START}}
}
`,
    peripheralTemplates: {} as any // Simplified for brevity
  },
  ti_tms320: {
    family: 'ti_tms320',
    vendor: 'Texas Instruments',
    halName: 'DriverLib',
    version: '2.0.0',
    includes: ['driverlib.h', 'device.h'],
    initCode: `
void main(void) {
  Device_init();
  Device_initGPIO();

  /* Initialize all configured peripherals */
  {{PERIPHERAL_INIT}}

  /* Enable global interrupts */
  EINT;
  ERTM;

  for(;;) {
    {{MAIN_LOOP}}
  }
}
`,
    peripheralTemplates: {} as any
  },
  infineon_aurix: {
    family: 'infineon_aurix',
    vendor: 'Infineon',
    halName: 'iLLD',
    version: '1.0.0',
    includes: ['Ifx_Types.h', 'IfxCpu.h'],
    initCode: `
int core0_main(void) {
  IfxCpu_enableInterrupts();

  /* Initialize all configured peripherals */
  {{PERIPHERAL_INIT}}

  while(1) {
    {{MAIN_LOOP}}
  }
  return 0;
}
`,
    peripheralTemplates: {} as any
  },
  nordic_nrf: {
    family: 'nordic_nrf',
    vendor: 'Nordic Semiconductor',
    halName: 'nrfx',
    version: '3.0.0',
    includes: ['nrf.h', 'nrfx.h'],
    initCode: `
int main(void) {
  /* Initialize board support */
  bsp_board_init(BSP_INIT_LEDS);

  /* Initialize all configured peripherals */
  {{PERIPHERAL_INIT}}

  for (;;) {
    {{MAIN_LOOP}}
  }
}
`,
    peripheralTemplates: {} as any
  },
  rpi_pico: {
    family: 'rpi_pico',
    vendor: 'Raspberry Pi',
    halName: 'Pico SDK',
    version: '1.5.0',
    includes: ['pico/stdlib.h', 'hardware/gpio.h'],
    initCode: `
int main() {
  stdio_init_all();

  /* Initialize all configured peripherals */
  {{PERIPHERAL_INIT}}

  while (true) {
    {{MAIN_LOOP}}
  }
  return 0;
}
`,
    peripheralTemplates: {} as any
  },
  nxp_imxrt: {
    family: 'nxp_imxrt',
    vendor: 'NXP',
    halName: 'MCUXpresso',
    version: '2.0.0',
    includes: ['fsl_common.h', 'pin_mux.h', 'clock_config.h'],
    initCode: `
int main(void) {
  BOARD_ConfigMPU();
  BOARD_InitBootPins();
  BOARD_InitBootClocks();
  BOARD_InitDebugConsole();

  /* Initialize all configured peripherals */
  {{PERIPHERAL_INIT}}

  while (1) {
    {{MAIN_LOOP}}
  }
}
`,
    peripheralTemplates: {} as any
  }
};

// ============================================================================
// RTOS Templates
// ============================================================================

const RTOS_TEMPLATES = {
  freertos: {
    includes: ['FreeRTOS.h', 'task.h', 'queue.h', 'semphr.h'],
    taskTemplate: `
void {{TASK_NAME}}_Task(void *pvParameters) {
  {{TASK_INIT}}

  for (;;) {
    {{TASK_BODY}}
    vTaskDelay(pdMS_TO_TICKS({{PERIOD_MS}}));
  }
}
`,
    createTask: `xTaskCreate({{TASK_NAME}}_Task, "{{TASK_NAME}}", {{STACK_SIZE}}, NULL, {{PRIORITY}}, NULL);`,
    startScheduler: 'vTaskStartScheduler();'
  },
  zephyr: {
    includes: ['zephyr/kernel.h', 'zephyr/device.h'],
    taskTemplate: `
K_THREAD_STACK_DEFINE({{TASK_NAME}}_stack, {{STACK_SIZE}});
struct k_thread {{TASK_NAME}}_thread_data;

void {{TASK_NAME}}_entry(void *p1, void *p2, void *p3) {
  {{TASK_INIT}}

  while (1) {
    {{TASK_BODY}}
    k_msleep({{PERIOD_MS}});
  }
}
`,
    createTask: `k_thread_create(&{{TASK_NAME}}_thread_data, {{TASK_NAME}}_stack, K_THREAD_STACK_SIZEOF({{TASK_NAME}}_stack), {{TASK_NAME}}_entry, NULL, NULL, NULL, {{PRIORITY}}, 0, K_NO_WAIT);`,
    startScheduler: '' // Zephyr starts automatically
  },
  tirtos: {
    includes: ['ti/sysbios/BIOS.h', 'ti/sysbios/knl/Task.h'],
    taskTemplate: `
void {{TASK_NAME}}_Fxn(UArg arg0, UArg arg1) {
  {{TASK_INIT}}

  for (;;) {
    {{TASK_BODY}}
    Task_sleep({{PERIOD_TICKS}});
  }
}
`,
    createTask: `Task_create({{TASK_NAME}}_Fxn, &taskParams, &eb);`,
    startScheduler: 'BIOS_start();'
  },
  autosar: {
    includes: ['Os.h', 'Os_Cfg.h'],
    taskTemplate: `
TASK({{TASK_NAME}}) {
  {{TASK_INIT}}
  {{TASK_BODY}}
  TerminateTask();
}
`,
    createTask: '',
    startScheduler: 'StartOS(OSDEFAULTAPPMODE);'
  }
};

// ============================================================================
// Firmware Generator
// ============================================================================

export class FirmwareGenerator extends EventEmitter {
  private config: FirmwareGeneratorConfig;

  constructor(config: Partial<FirmwareGeneratorConfig> = {}) {
    super();
    this.config = {
      templatesPath: config.templatesPath || './templates/firmware',
      outputPath: config.outputPath || './output/firmware',
      enableAIAssist: config.enableAIAssist !== false,
      supportedFamilies: config.supportedFamilies || ['stm32', 'esp32', 'ti_tms320', 'infineon_aurix', 'nordic_nrf', 'rpi_pico', 'nxp_imxrt'],
      defaultRTOS: config.defaultRTOS || 'freertos'
    };
  }

  /**
   * Generate a complete firmware project
   */
  async generate(requirements: FirmwareRequirements): Promise<GenerationResult> {
    const startTime = Date.now();
    const warnings: string[] = [];
    const errors: string[] = [];
    const generatedFiles: GeneratedFile[] = [];

    try {
      this.emit('generation:start', { projectName: requirements.projectName });
      logger.info('Starting firmware generation', {
        projectName: requirements.projectName,
        mcu: requirements.targetMcu.family
      });

      // Validate MCU support
      if (!this.config.supportedFamilies.includes(requirements.targetMcu.family)) {
        throw new ServiceError(
          `Unsupported MCU family: ${requirements.targetMcu.family}`,
          ErrorCodes.VALIDATION_ERROR,
          { supportedFamilies: this.config.supportedFamilies }
        );
      }

      const mcuConfig = MCU_CONFIGS[requirements.targetMcu.family];

      // Step 1: Generate project structure
      this.emit('generation:progress', { phase: 'structure', progress: 10 });
      const structureFiles = this.generateProjectStructure(requirements);
      generatedFiles.push(...structureFiles);

      // Step 2: Generate HAL configuration
      this.emit('generation:progress', { phase: 'hal', progress: 30 });
      const halFiles = this.generateHALCode(requirements, mcuConfig);
      generatedFiles.push(...halFiles);

      // Step 3: Generate peripheral drivers
      this.emit('generation:progress', { phase: 'drivers', progress: 50 });
      const driverFiles = this.generateDrivers(requirements);
      generatedFiles.push(...driverFiles);

      // Step 4: Generate RTOS configuration (if enabled)
      if (requirements.rtos) {
        this.emit('generation:progress', { phase: 'rtos', progress: 65 });
        const rtosFiles = this.generateRTOSConfig(requirements);
        generatedFiles.push(...rtosFiles);
      }

      // Step 5: Generate main application code
      this.emit('generation:progress', { phase: 'application', progress: 80 });
      const appFiles = this.generateApplicationCode(requirements, mcuConfig);
      generatedFiles.push(...appFiles);

      // Step 6: Generate build configuration
      this.emit('generation:progress', { phase: 'build', progress: 90 });
      const buildFiles = this.generateBuildConfig(requirements);
      generatedFiles.push(...buildFiles);

      // Create firmware project object
      const firmwareProject: FirmwareProject = {
        id: uuidv4(),
        projectId: uuidv4(),
        name: requirements.projectName,
        targetMcu: requirements.targetMcu,
        rtos: requirements.rtos,
        hal: {
          type: 'vendor',
          peripherals: requirements.peripherals.map(p => ({
            type: p.type,
            instance: p.instance,
            config: p.config,
            pinMapping: []
          }))
        },
        drivers: this.buildDriverList(requirements),
        tasks: requirements.rtos ? this.buildTaskList(requirements) : [],
        buildConfig: requirements.buildConfig || this.getDefaultBuildConfig(requirements.targetMcu.family),
        generatedFiles,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      this.emit('generation:progress', { phase: 'complete', progress: 100 });
      this.emit('generation:complete', { firmwareProject });

      logger.info('Firmware generation complete', {
        projectName: requirements.projectName,
        fileCount: generatedFiles.length,
        duration: Date.now() - startTime
      });

      return {
        success: true,
        firmwareProject,
        generatedFiles,
        buildInstructions: this.generateBuildInstructions(requirements),
        warnings,
        errors,
        generationTime: Date.now() - startTime
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      errors.push(errorMessage);
      logger.error('Firmware generation failed', { error: errorMessage });

      this.emit('generation:error', { error: errorMessage });

      return {
        success: false,
        generatedFiles,
        buildInstructions: '',
        warnings,
        errors,
        generationTime: Date.now() - startTime
      };
    }
  }

  /**
   * Generate project structure files
   */
  private generateProjectStructure(requirements: FirmwareRequirements): GeneratedFile[] {
    const files: GeneratedFile[] = [];
    const projectName = requirements.projectName.replace(/\s+/g, '_').toLowerCase();

    // README.md
    files.push({
      path: 'README.md',
      type: 'config',
      content: `# ${requirements.projectName}

## Overview
Auto-generated firmware project for ${requirements.targetMcu.family.toUpperCase()} (${requirements.targetMcu.part})

## Target MCU
- Family: ${requirements.targetMcu.family}
- Part: ${requirements.targetMcu.part}
- Core: ${requirements.targetMcu.core}
- Flash: ${requirements.targetMcu.flashSize / 1024}KB
- RAM: ${requirements.targetMcu.ramSize / 1024}KB
- Clock: ${requirements.targetMcu.clockSpeed / 1e6}MHz

## Build Instructions
\`\`\`bash
mkdir build && cd build
cmake ..
make
\`\`\`

## Generated by
Nexus EE Design Partner - Firmware Generator
Generated: ${new Date().toISOString()}
`,
      generatedAt: new Date().toISOString()
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
      generatedAt: new Date().toISOString()
    });

    return files;
  }

  /**
   * Generate HAL configuration code
   */
  private generateHALCode(requirements: FirmwareRequirements, mcuConfig: HALTemplate): GeneratedFile[] {
    const files: GeneratedFile[] = [];

    // Generate main HAL header
    const halIncludes = mcuConfig.includes.map(inc => `#include "${inc}"`).join('\n');

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
${halIncludes}

/* Exported types */
/* Exported constants */
/* Exported macro */
/* Exported functions */
void Error_Handler(void);

/* Private defines */
${this.generatePinDefines(requirements)}

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
`,
      generatedAt: new Date().toISOString()
    });

    // Generate peripheral initialization functions
    for (const peripheral of requirements.peripherals) {
      const template = mcuConfig.peripheralTemplates[peripheral.type];
      if (template) {
        files.push({
          path: `Core/Src/${peripheral.type}_${peripheral.instance.toLowerCase()}.c`,
          type: 'source',
          content: this.generatePeripheralInitCode(peripheral, template),
          generatedAt: new Date().toISOString()
        });
      }
    }

    return files;
  }

  /**
   * Generate driver code
   */
  private generateDrivers(requirements: FirmwareRequirements): GeneratedFile[] {
    const files: GeneratedFile[] = [];

    for (const peripheral of requirements.peripherals) {
      if (peripheral.connectedTo) {
        // Generate driver for connected component
        files.push({
          path: `Drivers/Inc/${peripheral.connectedTo.toLowerCase()}_driver.h`,
          type: 'header',
          content: this.generateDriverHeader(peripheral),
          generatedAt: new Date().toISOString()
        });

        files.push({
          path: `Drivers/Src/${peripheral.connectedTo.toLowerCase()}_driver.c`,
          type: 'source',
          content: this.generateDriverSource(peripheral),
          generatedAt: new Date().toISOString()
        });
      }
    }

    return files;
  }

  /**
   * Generate RTOS configuration
   */
  private generateRTOSConfig(requirements: FirmwareRequirements): GeneratedFile[] {
    const files: GeneratedFile[] = [];
    const rtos = requirements.rtos!;
    const rtosTemplate = RTOS_TEMPLATES[rtos.type as keyof typeof RTOS_TEMPLATES];

    if (!rtosTemplate) {
      return files;
    }

    // FreeRTOSConfig.h or equivalent
    if (rtos.type === 'freertos') {
      files.push({
        path: 'Core/Inc/FreeRTOSConfig.h',
        type: 'header',
        content: `/**
 * @file FreeRTOSConfig.h
 * @brief FreeRTOS configuration
 * @generated Nexus EE Design Partner
 */

#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H

#define configUSE_PREEMPTION                    1
#define configUSE_PORT_OPTIMISED_TASK_SELECTION 0
#define configUSE_TICKLESS_IDLE                 0
#define configCPU_CLOCK_HZ                      ${requirements.targetMcu.clockSpeed}
#define configTICK_RATE_HZ                      ${rtos.tickRate}
#define configMAX_PRIORITIES                    ${Math.min(rtos.maxTasks, 7)}
#define configMINIMAL_STACK_SIZE                128
#define configTOTAL_HEAP_SIZE                   ${rtos.heapSize}
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
`,
        generatedAt: new Date().toISOString()
      });
    }

    // Generate task files
    files.push({
      path: 'Core/Src/app_tasks.c',
      type: 'source',
      content: this.generateTasksCode(requirements, rtosTemplate),
      generatedAt: new Date().toISOString()
    });

    return files;
  }

  /**
   * Generate main application code
   */
  private generateApplicationCode(requirements: FirmwareRequirements, mcuConfig: HALTemplate): GeneratedFile[] {
    const files: GeneratedFile[] = [];

    // Generate peripheral init calls
    const peripheralInit = requirements.peripherals
      .map(p => `  MX_${p.type.toUpperCase()}${p.instance}_Init();`)
      .join('\n');

    // Generate RTOS start code if enabled
    const rtosStart = requirements.rtos
      ? RTOS_TEMPLATES[requirements.rtos.type as keyof typeof RTOS_TEMPLATES]?.startScheduler || ''
      : '';

    // Generate main loop code
    const mainLoop = requirements.rtos ? '/* RTOS running */' : '/* Main loop */';

    const mainCode = mcuConfig.initCode
      .replace('{{PERIPHERAL_INIT}}', peripheralInit)
      .replace('{{RTOS_START}}', rtosStart)
      .replace('{{MAIN_LOOP}}', mainLoop);

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
${requirements.rtos ? '#include "FreeRTOS.h"\n#include "task.h"' : ''}

/* Private function prototypes */
${mcuConfig.initCode.includes('SystemClock_Config') ? 'void SystemClock_Config(void);' : ''}
${requirements.peripherals.map(p => `void MX_${p.type.toUpperCase()}${p.instance}_Init(void);`).join('\n')}

${mainCode}

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
      generatedAt: new Date().toISOString()
    });

    return files;
  }

  /**
   * Generate build configuration files
   */
  private generateBuildConfig(requirements: FirmwareRequirements): GeneratedFile[] {
    const files: GeneratedFile[] = [];
    const projectName = requirements.projectName.replace(/\s+/g, '_').toLowerCase();
    const buildConfig = requirements.buildConfig || this.getDefaultBuildConfig(requirements.targetMcu.family);

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
set(CMAKE_C_FLAGS_DEBUG "-g -${buildConfig.debugSymbols ? 'Og' : 'O0'}")
set(CMAKE_C_FLAGS_RELEASE "-${buildConfig.optimizationLevel}")

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS "\${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -specs=nosys.specs")

# Include directories
include_directories(
  Core/Inc
  Drivers/Inc
  ${requirements.rtos ? 'Middlewares/Third_Party/FreeRTOS/Source/include\n  Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1' : ''}
)

# Source files
file(GLOB_RECURSE SOURCES
  "Core/Src/*.c"
  "Drivers/Src/*.c"
  ${requirements.rtos ? '"Middlewares/Third_Party/FreeRTOS/Source/*.c"\n  "Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1/*.c"' : ''}
)

# Startup file
set(STARTUP_FILE startup_stm32h755xx.s)

# Linker script
set(LINKER_SCRIPT STM32H755ZITx_FLASH.ld)

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
      generatedAt: new Date().toISOString()
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

C_DEFS = ${Object.entries(buildConfig.defines || {}).map(([k, v]) => `-D${k}=${v}`).join(' ')}

OPT = -${buildConfig.optimizationLevel}

CFLAGS = $(MCU) $(C_DEFS) $(C_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections
${buildConfig.debugSymbols ? 'CFLAGS += -g -gdwarf-2' : ''}

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
      generatedAt: new Date().toISOString()
    });

    return files;
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  private generatePinDefines(requirements: FirmwareRequirements): string {
    const defines: string[] = [];

    for (const peripheral of requirements.peripherals) {
      if (peripheral.config.pins) {
        const pins = peripheral.config.pins as Array<{ name: string; pin: string }>;
        for (const pin of pins) {
          defines.push(`#define ${peripheral.instance}_${pin.name}_Pin ${pin.pin}`);
        }
      }
    }

    return defines.join('\n');
  }

  private generatePeripheralInitCode(peripheral: PeripheralRequirement, template: PeripheralTemplate): string {
    const instanceUpper = peripheral.instance.toUpperCase();
    const instanceNum = peripheral.instance.replace(/\D/g, '');

    let code = template.initTemplate
      .replace(/{{N}}/g, instanceNum)
      .replace('{{' + peripheral.type.toUpperCase() + '_CONFIG}}', template.configStruct);

    // Replace config placeholders
    for (const [key, value] of Object.entries(peripheral.config)) {
      code = code.replace(new RegExp(`{{${key.toUpperCase()}}}`, 'g'), String(value));
    }

    return `/**
 * @file ${peripheral.type}_${peripheral.instance.toLowerCase()}.c
 * @brief ${peripheral.type.toUpperCase()} ${peripheral.instance} initialization
 * @generated Nexus EE Design Partner
 */

#include "main.h"

${code}
`;
  }

  private generateDriverHeader(peripheral: PeripheralRequirement): string {
    const componentName = peripheral.connectedTo!.replace(/\d+$/, '');
    const guardName = `__${componentName.toUpperCase()}_DRIVER_H`;

    return `/**
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

/* Type definitions */
typedef struct {
  uint8_t initialized;
  void* handle;
} ${componentName}_Handle_t;

/* Function prototypes */
int ${componentName}_Init(${componentName}_Handle_t* handle);
int ${componentName}_DeInit(${componentName}_Handle_t* handle);
int ${componentName}_Read(${componentName}_Handle_t* handle, uint8_t* data, uint16_t len);
int ${componentName}_Write(${componentName}_Handle_t* handle, const uint8_t* data, uint16_t len);

#ifdef __cplusplus
}
#endif

#endif /* ${guardName} */
`;
  }

  private generateDriverSource(peripheral: PeripheralRequirement): string {
    const componentName = peripheral.connectedTo!.replace(/\d+$/, '');

    return `/**
 * @file ${peripheral.connectedTo!.toLowerCase()}_driver.c
 * @brief Driver implementation for ${peripheral.connectedTo}
 * @generated Nexus EE Design Partner
 */

#include "${peripheral.connectedTo!.toLowerCase()}_driver.h"

int ${componentName}_Init(${componentName}_Handle_t* handle) {
  if (handle == NULL) {
    return -1;
  }

  /* Initialize hardware */
  // TODO: Add initialization code

  handle->initialized = 1;
  return 0;
}

int ${componentName}_DeInit(${componentName}_Handle_t* handle) {
  if (handle == NULL || !handle->initialized) {
    return -1;
  }

  /* Deinitialize hardware */
  // TODO: Add deinitialization code

  handle->initialized = 0;
  return 0;
}

int ${componentName}_Read(${componentName}_Handle_t* handle, uint8_t* data, uint16_t len) {
  if (handle == NULL || !handle->initialized || data == NULL) {
    return -1;
  }

  /* Read data */
  // TODO: Add read implementation

  return 0;
}

int ${componentName}_Write(${componentName}_Handle_t* handle, const uint8_t* data, uint16_t len) {
  if (handle == NULL || !handle->initialized || data == NULL) {
    return -1;
  }

  /* Write data */
  // TODO: Add write implementation

  return 0;
}
`;
  }

  private generateTasksCode(requirements: FirmwareRequirements, rtosTemplate: typeof RTOS_TEMPLATES.freertos): string {
    const tasks = this.buildTaskList(requirements);

    const taskFunctions = tasks.map(task => {
      return rtosTemplate.taskTemplate
        .replace(/{{TASK_NAME}}/g, task.name)
        .replace('{{TASK_INIT}}', '/* Task initialization */')
        .replace('{{TASK_BODY}}', '/* Task body */')
        .replace('{{PERIOD_MS}}', String(task.period || 100))
        .replace('{{STACK_SIZE}}', String(task.stackSize))
        .replace('{{PRIORITY}}', String(task.priority));
    }).join('\n');

    const taskCreation = tasks.map(task => {
      return rtosTemplate.createTask
        .replace(/{{TASK_NAME}}/g, task.name)
        .replace('{{STACK_SIZE}}', String(task.stackSize))
        .replace('{{PRIORITY}}', String(task.priority));
    }).join('\n  ');

    return `/**
 * @file app_tasks.c
 * @brief RTOS task implementations
 * @generated Nexus EE Design Partner
 */

#include "main.h"
#include "FreeRTOS.h"
#include "task.h"

/* Task functions */
${taskFunctions}

/* Create all tasks */
void App_CreateTasks(void) {
  ${taskCreation}
}
`;
  }

  private buildDriverList(requirements: FirmwareRequirements): Driver[] {
    return requirements.peripherals
      .filter(p => p.connectedTo)
      .map(p => ({
        id: uuidv4(),
        name: `${p.connectedTo}_Driver`,
        type: p.connectedTo!.replace(/\d+$/, ''),
        component: p.connectedTo!,
        interface: p.type,
        functions: [
          { name: 'Init', description: 'Initialize driver', parameters: [], returnType: 'int' },
          { name: 'DeInit', description: 'Deinitialize driver', parameters: [], returnType: 'int' },
          { name: 'Read', description: 'Read data', parameters: [
              { name: 'data', type: 'uint8_t*', description: 'Data buffer' },
              { name: 'len', type: 'uint16_t', description: 'Buffer length' }
            ], returnType: 'int' },
          { name: 'Write', description: 'Write data', parameters: [
              { name: 'data', type: 'const uint8_t*', description: 'Data buffer' },
              { name: 'len', type: 'uint16_t', description: 'Data length' }
            ], returnType: 'int' }
        ]
      }));
  }

  private buildTaskList(requirements: FirmwareRequirements): FirmwareTask[] {
    const tasks: FirmwareTask[] = [];

    // Main task
    tasks.push({
      name: 'MainTask',
      priority: 2,
      stackSize: 512,
      period: 100,
      description: 'Main application task'
    });

    // Communication task if UART/SPI/I2C present
    const hasComm = requirements.peripherals.some(p =>
      ['uart', 'spi', 'i2c', 'can'].includes(p.type)
    );
    if (hasComm) {
      tasks.push({
        name: 'CommTask',
        priority: 3,
        stackSize: 512,
        period: 10,
        description: 'Communication handling task'
      });
    }

    // Sensor task if ADC present
    const hasADC = requirements.peripherals.some(p => p.type === 'adc');
    if (hasADC) {
      tasks.push({
        name: 'SensorTask',
        priority: 2,
        stackSize: 256,
        period: 50,
        description: 'Sensor reading task'
      });
    }

    return tasks;
  }

  private getDefaultBuildConfig(family: MCUFamily): BuildConfig {
    return {
      toolchain: 'gcc-arm',
      buildSystem: 'cmake',
      optimizationLevel: 'O2',
      debugSymbols: true,
      defines: {
        'USE_HAL_DRIVER': '1',
        [`STM32H755xx`]: '1'
      }
    };
  }

  private generateBuildInstructions(requirements: FirmwareRequirements): string {
    return `# Build Instructions for ${requirements.projectName}

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
openocd -f interface/stlink.cfg -f target/stm32h7x.cfg -c "program build/${requirements.projectName.replace(/\s+/g, '_').toLowerCase()}.elf verify reset exit"

# Using ST-Link
st-flash write build/${requirements.projectName.replace(/\s+/g, '_').toLowerCase()}.bin 0x08000000
\`\`\`
`;
  }
}

export default FirmwareGenerator;
