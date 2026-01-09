/**
 * EE Design Partner - Firmware Generation Prompts
 *
 * Production-ready prompts for AI-assisted firmware development:
 * - Firmware architecture planning
 * - HAL code generation
 * - Driver code generation
 * - Code validation and review
 */

import {
  LLMMessage,
  FirmwareArchitecture,
  FirmwareLayer,
  FirmwareModule,
  FirmwareTaskSpec,
  FirmwareInterface,
  CodeReviewResult,
  CodeIssue,
} from '../types.js';

// ============================================================================
// Types
// ============================================================================

export type MCUFamily =
  | 'stm32'
  | 'esp32'
  | 'ti_tms320'
  | 'infineon_aurix'
  | 'nordic_nrf'
  | 'rpi_pico'
  | 'nxp_imxrt';

export type RTOSType = 'freertos' | 'zephyr' | 'tirtos' | 'autosar' | 'bare_metal';

export interface FirmwareRequirements {
  projectName: string;
  description: string;
  mcuFamily: MCUFamily;
  mcuPart: string;
  clockSpeed: number;
  flashSize: number;
  ramSize: number;
  rtos?: RTOSType;
  peripherals: PeripheralRequirement[];
  features: string[];
  safetyLevel?: 'none' | 'asil_a' | 'asil_b' | 'asil_c' | 'asil_d' | 'sil_2' | 'sil_3';
}

export interface PeripheralRequirement {
  type: string;
  instance: string;
  purpose: string;
  config: Record<string, unknown>;
}

export interface PeripheralSpec {
  type: 'uart' | 'spi' | 'i2c' | 'adc' | 'dac' | 'pwm' | 'timer' | 'can' | 'usb' | 'gpio';
  instance: string;
  config: {
    baudRate?: number;
    clockSpeed?: number;
    mode?: string;
    bits?: number;
    channels?: number[];
    resolution?: number;
    frequency?: number;
    [key: string]: unknown;
  };
  pins: Array<{
    signal: string;
    port: string;
    pin: number;
    alternate?: number;
  }>;
}

export interface ComponentDatasheet {
  name: string;
  manufacturer: string;
  partNumber: string;
  interface: 'i2c' | 'spi' | 'uart' | 'gpio' | 'analog';
  registers?: Array<{
    address: number | string;
    name: string;
    description: string;
    fields?: Array<{
      name: string;
      bits: string;
      description: string;
    }>;
  }>;
  commands?: Array<{
    name: string;
    opcode: number | string;
    description: string;
    parameters?: string[];
  }>;
  typicalInit?: string[];
  notes?: string[];
}

export type CodeLanguage = 'c' | 'cpp' | 'rust';

export interface CodeForReview {
  filename: string;
  language: CodeLanguage;
  content: string;
  context?: string;
  checkFor?: ('safety' | 'performance' | 'style' | 'security' | 'memory')[];
}

// ============================================================================
// Firmware Architecture Prompt
// ============================================================================

/**
 * Generate a prompt for planning firmware architecture
 */
export function generateFirmwareArchitecturePrompt(
  requirements: FirmwareRequirements,
  mcu: { family: MCUFamily; part: string },
  rtos: RTOSType
): LLMMessage[] {
  const systemPrompt = `You are a senior embedded systems architect with expertise in:
- Real-time operating systems (FreeRTOS, Zephyr, TI-RTOS, AUTOSAR)
- MCU families (STM32, ESP32, TI C2000, Infineon AURIX, Nordic nRF)
- Safety-critical systems (ISO 26262, IEC 61508)
- Embedded software design patterns
- Hardware abstraction and portability

Your task is to design a clean, maintainable firmware architecture following best practices:
1. Proper layering (HAL -> Driver -> Service -> Application)
2. Clear module boundaries and interfaces
3. Appropriate task decomposition for RTOS
4. Memory-efficient design
5. Testability and debugging support
6. Safety considerations where applicable

You must respond with a valid JSON object containing the architecture design.`;

  const userPrompt = `Design a firmware architecture for this project:

**Project:** ${requirements.projectName}
**Description:** ${requirements.description}

**Target Hardware:**
- MCU Family: ${mcu.family.toUpperCase()}
- Part Number: ${mcu.part}
- Clock: ${requirements.clockSpeed / 1e6} MHz
- Flash: ${requirements.flashSize / 1024} KB
- RAM: ${requirements.ramSize / 1024} KB

**Operating System:** ${rtos === 'bare_metal' ? 'Bare Metal (super loop)' : rtos.toUpperCase()}

**Peripherals Required:**
${requirements.peripherals.map(p => `- ${p.type.toUpperCase()} ${p.instance}: ${p.purpose}`).join('\n')}

**Features:**
${requirements.features.map(f => `- ${f}`).join('\n')}

${requirements.safetyLevel && requirements.safetyLevel !== 'none' ? `**Safety Level:** ${requirements.safetyLevel.toUpperCase()} - Include appropriate safety mechanisms` : ''}

Please provide the architecture in this JSON format:
{
  "layers": [
    {
      "name": "string - layer name like HAL, Driver, Service, Application",
      "description": "string - what this layer does",
      "modules": ["list of module names in this layer"]
    }
  ],
  "modules": [
    {
      "name": "string - module name",
      "layer": "string - which layer",
      "description": "string - module purpose",
      "files": ["list of source files like module.c, module.h"],
      "dependencies": ["list of modules this depends on"],
      "publicApi": ["list of public function names"]
    }
  ],
  "tasks": [
    {
      "name": "string - task name",
      "priority": number (1-10, higher = more important),
      "stackSize": number (bytes),
      "periodMs": number (0 for event-driven),
      "description": "string - what this task does",
      "modules": ["modules this task uses"]
    }
  ],
  "interfaces": [
    {
      "name": "string - interface name like UART_DEBUG, SPI_FLASH",
      "type": "uart|spi|i2c|can|usb|ethernet|gpio",
      "peripheral": "string - peripheral instance like USART1",
      "config": { "key": "value pairs for configuration" }
    }
  ],
  "dependencies": ["external libraries needed like FreeRTOS, CMSIS"],
  "memoryMap": {
    "flashUsageEstimate": number (bytes),
    "ramUsageEstimate": number (bytes),
    "heapSize": number (bytes for RTOS heap),
    "stackTotal": number (bytes for all task stacks)
  },
  "notes": ["important design decisions and rationale"]
}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

/**
 * Validator for firmware architecture response
 */
export function validateFirmwareArchitectureResponse(response: string): FirmwareArchitecture | null {
  try {
    const parsed = JSON.parse(response);

    // Validate required fields
    if (!Array.isArray(parsed.layers) || parsed.layers.length === 0) {
      return null;
    }
    if (!Array.isArray(parsed.modules) || parsed.modules.length === 0) {
      return null;
    }

    // Validate layer structure
    const layers: FirmwareLayer[] = parsed.layers.map((l: unknown) => {
      const layer = l as Record<string, unknown>;
      return {
        name: String(layer.name || ''),
        description: String(layer.description || ''),
        modules: Array.isArray(layer.modules) ? layer.modules.map(String) : [],
      };
    });

    // Validate module structure
    const modules: FirmwareModule[] = parsed.modules.map((m: unknown) => {
      const mod = m as Record<string, unknown>;
      return {
        name: String(mod.name || ''),
        layer: String(mod.layer || ''),
        description: String(mod.description || ''),
        files: Array.isArray(mod.files) ? mod.files.map(String) : [],
        dependencies: Array.isArray(mod.dependencies) ? mod.dependencies.map(String) : [],
        publicApi: Array.isArray(mod.publicApi) ? mod.publicApi.map(String) : [],
      };
    });

    // Validate tasks
    const tasks: FirmwareTaskSpec[] = Array.isArray(parsed.tasks)
      ? parsed.tasks.map((t: unknown) => {
          const task = t as Record<string, unknown>;
          return {
            name: String(task.name || ''),
            priority: Number(task.priority) || 1,
            stackSize: Number(task.stackSize) || 512,
            periodMs: Number(task.periodMs) || 0,
            description: String(task.description || ''),
            modules: Array.isArray(task.modules) ? task.modules.map(String) : [],
          };
        })
      : [];

    // Validate interfaces
    const interfaces: FirmwareInterface[] = Array.isArray(parsed.interfaces)
      ? parsed.interfaces.map((i: unknown) => {
          const iface = i as Record<string, unknown>;
          return {
            name: String(iface.name || ''),
            type: iface.type as FirmwareInterface['type'],
            peripheral: String(iface.peripheral || ''),
            config: (iface.config as Record<string, unknown>) || {},
          };
        })
      : [];

    return {
      layers,
      modules,
      tasks,
      interfaces,
      dependencies: Array.isArray(parsed.dependencies) ? parsed.dependencies.map(String) : [],
    };
  } catch {
    return null;
  }
}

// ============================================================================
// HAL Code Generation Prompt
// ============================================================================

/**
 * Generate a prompt for creating HAL code for a peripheral
 */
export function generateHALCodePrompt(peripheral: PeripheralSpec, mcu: MCUFamily): LLMMessage[] {
  const systemPrompt = `You are an expert embedded systems developer specializing in hardware abstraction layers.

Your task is to generate production-quality HAL code for the specified peripheral on ${mcu.toUpperCase()} MCUs.

Code requirements:
1. Follow ${getMCUCodingStandard(mcu)} coding standards
2. Use the official ${getMCUHALName(mcu)} HAL/driver library
3. Include complete error handling
4. Add comprehensive documentation (Doxygen format)
5. Make code thread-safe if applicable
6. Include initialization, configuration, and runtime functions
7. Provide both blocking and interrupt-driven variants where applicable

You must respond with valid JSON containing the generated code files.`;

  const userPrompt = `Generate HAL code for this peripheral:

**Peripheral:** ${peripheral.type.toUpperCase()} ${peripheral.instance}

**Configuration:**
${Object.entries(peripheral.config)
  .map(([key, value]) => `- ${key}: ${value}`)
  .join('\n')}

**Pin Mapping:**
${peripheral.pins.map(p => `- ${p.signal}: ${p.port}${p.pin}${p.alternate !== undefined ? ` (AF${p.alternate})` : ''}`).join('\n')}

**MCU:** ${mcu.toUpperCase()}

Please provide the code in this JSON format:
{
  "headerFile": {
    "filename": "string - like hal_uart.h",
    "content": "string - complete header file content"
  },
  "sourceFile": {
    "filename": "string - like hal_uart.c",
    "content": "string - complete source file content"
  },
  "usage": "string - example usage code",
  "dependencies": ["list of required header files"],
  "notes": ["any important implementation notes"]
}

The generated code should include:
1. Initialization function
2. Configuration function
3. Enable/Disable functions
4. Data transfer functions (read/write)
5. Status query functions
6. Interrupt handlers if applicable
7. Error handling`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

// ============================================================================
// Driver Code Generation Prompt
// ============================================================================

/**
 * Generate a prompt for creating a driver for an external component
 */
export function generateDriverCodePrompt(
  component: ComponentDatasheet,
  datasheet: string
): LLMMessage[] {
  const systemPrompt = `You are an expert embedded driver developer who has written drivers for hundreds of components.

Your task is to generate a complete, production-ready driver for the specified component.

Driver requirements:
1. Clean, readable code following MISRA-C guidelines where practical
2. Complete abstraction from the underlying interface (I2C/SPI/etc)
3. All functions from the component datasheet implemented
4. Comprehensive error handling and status reporting
5. Thread-safe design with mutex protection where needed
6. Low-level register access and high-level convenience functions
7. Full Doxygen documentation
8. Example usage code

You must respond with valid JSON containing the driver code files.`;

  const userPrompt = `Generate a driver for this component:

**Component:** ${component.name}
**Manufacturer:** ${component.manufacturer}
**Part Number:** ${component.partNumber}
**Interface:** ${component.interface.toUpperCase()}

${component.registers?.length ? `**Key Registers:**
${component.registers.slice(0, 10).map(r => `- 0x${typeof r.address === 'number' ? r.address.toString(16).padStart(2, '0') : r.address}: ${r.name} - ${r.description}`).join('\n')}` : ''}

${component.commands?.length ? `**Commands:**
${component.commands.map(c => `- ${c.name} (0x${typeof c.opcode === 'number' ? c.opcode.toString(16) : c.opcode}): ${c.description}`).join('\n')}` : ''}

${component.typicalInit?.length ? `**Typical Initialization Sequence:**
${component.typicalInit.map((step, i) => `${i + 1}. ${step}`).join('\n')}` : ''}

**Additional Datasheet Information:**
${datasheet}

${component.notes?.length ? `**Notes:**
${component.notes.map(n => `- ${n}`).join('\n')}` : ''}

Please provide the driver in this JSON format:
{
  "headerFile": {
    "filename": "string - like drv_component.h",
    "content": "string - complete header with types, defines, function prototypes"
  },
  "sourceFile": {
    "filename": "string - like drv_component.c",
    "content": "string - complete implementation"
  },
  "types": {
    "filename": "string - like drv_component_types.h",
    "content": "string - type definitions, enums, structs"
  },
  "example": {
    "filename": "string - like example_component.c",
    "content": "string - usage example"
  },
  "testStub": {
    "filename": "string - like test_component.c",
    "content": "string - unit test template"
  },
  "dependencies": ["required headers/libraries"],
  "notes": ["implementation notes and gotchas"]
}

Include these driver functions:
1. Init/Deinit
2. Configuration functions
3. Read/Write functions
4. Status/diagnostic functions
5. Power management (if applicable)
6. Interrupt handling (if applicable)
7. Self-test function`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

// ============================================================================
// Code Validation Prompt
// ============================================================================

/**
 * Generate a prompt for validating firmware code
 */
export function validateCodePrompt(code: CodeForReview, language: CodeLanguage): LLMMessage[] {
  const checksToPerform = code.checkFor || ['safety', 'performance', 'style', 'memory'];

  const systemPrompt = `You are a senior embedded software engineer performing a thorough code review.

Your expertise includes:
1. MISRA-C/C++ compliance and safety-critical coding
2. Memory management in resource-constrained systems
3. Real-time performance optimization
4. Security vulnerabilities in embedded systems
5. Best practices for ${language.toUpperCase()} embedded development

Review Focus: ${checksToPerform.join(', ')}

Be specific and actionable in your feedback. Provide line numbers where possible.
Consider the embedded/real-time context - heap fragmentation, stack usage, interrupt safety, etc.

You must respond with a valid JSON object containing your review.`;

  const userPrompt = `Review this ${language.toUpperCase()} firmware code:

**File:** ${code.filename}

${code.context ? `**Context:** ${code.context}\n` : ''}

\`\`\`${language}
${code.content}
\`\`\`

Please provide your review in this JSON format:
{
  "score": number (0-100),
  "passed": boolean (true if score >= 70 and no critical issues),
  "summary": "string - overall assessment",
  "issues": [
    {
      "severity": "critical|error|warning|info",
      "line": number (if applicable),
      "file": "string - filename",
      "message": "string - what's wrong",
      "suggestion": "string - how to fix it"
    }
  ],
  "suggestions": ["general improvement suggestions"],
  "securityConcerns": ["any security-related issues"],
  "performanceNotes": ["performance observations"],
  "memoryAnalysis": {
    "stackUsage": "string - assessment",
    "heapUsage": "string - assessment",
    "staticAllocation": "string - assessment",
    "concerns": ["specific memory concerns"]
  },
  "misraViolations": ["MISRA rule violations if any"],
  "positives": ["things done well"]
}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

/**
 * Validator for code review response
 */
export function validateCodeReviewResponse(response: string): CodeReviewResult | null {
  try {
    const parsed = JSON.parse(response);

    if (typeof parsed.score !== 'number' || typeof parsed.passed !== 'boolean') {
      return null;
    }

    const issues: CodeIssue[] = Array.isArray(parsed.issues)
      ? parsed.issues.map((i: unknown) => {
          const issue = i as Record<string, unknown>;
          return {
            severity: (issue.severity as CodeIssue['severity']) || 'warning',
            line: typeof issue.line === 'number' ? issue.line : undefined,
            file: typeof issue.file === 'string' ? issue.file : undefined,
            message: String(issue.message || ''),
            suggestion: String(issue.suggestion || ''),
          };
        })
      : [];

    return {
      score: parsed.score,
      passed: parsed.passed,
      issues,
      suggestions: Array.isArray(parsed.suggestions) ? parsed.suggestions.map(String) : [],
      securityConcerns: Array.isArray(parsed.securityConcerns)
        ? parsed.securityConcerns.map(String)
        : [],
      performanceNotes: Array.isArray(parsed.performanceNotes)
        ? parsed.performanceNotes.map(String)
        : [],
    };
  } catch {
    return null;
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

function getMCUCodingStandard(mcu: MCUFamily): string {
  const standards: Record<MCUFamily, string> = {
    stm32: 'STM32Cube/CMSIS',
    esp32: 'ESP-IDF',
    ti_tms320: 'TI C2000 DriverLib',
    infineon_aurix: 'iLLD (Infineon Low Level Driver)',
    nordic_nrf: 'nrfx/Zephyr',
    rpi_pico: 'Pico SDK',
    nxp_imxrt: 'MCUXpresso SDK',
  };
  return standards[mcu] || 'vendor-standard';
}

function getMCUHALName(mcu: MCUFamily): string {
  const halNames: Record<MCUFamily, string> = {
    stm32: 'STM32 HAL',
    esp32: 'ESP-IDF Driver',
    ti_tms320: 'DriverLib',
    infineon_aurix: 'iLLD',
    nordic_nrf: 'nrfx',
    rpi_pico: 'Pico SDK',
    nxp_imxrt: 'MCUXpresso',
  };
  return halNames[mcu] || 'vendor HAL';
}

// ============================================================================
// Additional Utility Prompts
// ============================================================================

/**
 * Generate prompt for creating interrupt service routines
 */
export function generateISRPrompt(
  peripheral: string,
  mcu: MCUFamily,
  requirements: string
): LLMMessage[] {
  return [
    {
      role: 'system',
      content: `You are an expert in writing interrupt service routines for embedded systems.

Focus on:
1. Minimal ISR execution time
2. Proper volatile usage
3. No blocking operations in ISR
4. Proper flag clearing
5. Thread-safe communication with main loop
6. ${getMCUHALName(mcu)} conventions`,
    },
    {
      role: 'user',
      content: `Generate ISR code for ${peripheral} on ${mcu.toUpperCase()}.

Requirements:
${requirements}

Provide complete ISR implementation with:
1. Vector handler function
2. Flag management
3. Data transfer to/from buffers
4. Error handling
5. Performance-critical sections marked`,
    },
  ];
}

/**
 * Generate prompt for state machine implementation
 */
export function generateStateMachinePrompt(
  name: string,
  states: string[],
  events: string[],
  description: string
): LLMMessage[] {
  return [
    {
      role: 'system',
      content: `You are an expert in implementing state machines for embedded systems.

Follow these patterns:
1. Table-driven state machine design
2. Clear state transition functions
3. Entry/exit actions for states
4. Event queue management
5. Timeout handling
6. Debug/trace capabilities`,
    },
    {
      role: 'user',
      content: `Generate a state machine implementation:

**Name:** ${name}
**Description:** ${description}

**States:**
${states.map(s => `- ${s}`).join('\n')}

**Events:**
${events.map(e => `- ${e}`).join('\n')}

Provide:
1. Header file with types and API
2. Source file with implementation
3. State transition table
4. Example usage`,
    },
  ];
}

// ============================================================================
// Exports
// ============================================================================

export default {
  generateFirmwareArchitecturePrompt,
  validateFirmwareArchitectureResponse,
  generateHALCodePrompt,
  generateDriverCodePrompt,
  validateCodePrompt,
  validateCodeReviewResponse,
  generateISRPrompt,
  generateStateMachinePrompt,
};
