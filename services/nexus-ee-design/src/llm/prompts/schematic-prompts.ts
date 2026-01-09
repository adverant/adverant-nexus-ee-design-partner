/**
 * EE Design Partner - Schematic Generation Prompts
 *
 * Production-ready prompts for AI-assisted schematic design:
 * - Block diagram generation
 * - Component selection
 * - Netlist generation
 * - Schematic validation
 */

import {
  LLMMessage,
  SchematicBlockDiagram,
  SchematicBlock,
  ComponentSelection,
  NetlistNode,
} from '../types.js';

// ============================================================================
// Types
// ============================================================================

export interface ProjectRequirements {
  name: string;
  description: string;
  targetMcu?: string;
  inputVoltage?: number | { min: number; max: number };
  outputVoltages?: Array<{ name: string; voltage: number; current: number }>;
  maxCurrent?: number;
  interfaces?: Array<{
    type: 'uart' | 'spi' | 'i2c' | 'can' | 'usb' | 'ethernet' | 'gpio' | 'adc' | 'pwm';
    count: number;
    config?: Record<string, unknown>;
  }>;
  features?: string[];
  constraints?: {
    maxComponents?: number;
    costTarget?: number;
    automotiveGrade?: boolean;
    industrialGrade?: boolean;
  };
}

export type ProjectType =
  | 'motor_controller'
  | 'power_supply'
  | 'sensor_node'
  | 'communication_gateway'
  | 'data_logger'
  | 'battery_management'
  | 'led_driver'
  | 'audio_amplifier'
  | 'rf_transceiver'
  | 'general';

export interface BlockRequirements {
  blockId: string;
  blockName: string;
  blockType: string;
  description: string;
  inputSignals: string[];
  outputSignals: string[];
  powerRails: string[];
  constraints?: {
    maxComponents?: number;
    preferredManufacturers?: string[];
    avoidManufacturers?: string[];
    tempRange?: { min: number; max: number };
    priceTarget?: number;
  };
}

export interface SchematicForValidation {
  name: string;
  components: Array<{
    reference: string;
    value: string;
    partNumber?: string;
    pins: string[];
  }>;
  nets: Array<{
    name: string;
    connections: string[];
    type?: 'power' | 'ground' | 'signal' | 'differential';
  }>;
  powerRails: Array<{
    name: string;
    voltage: number;
    maxCurrent?: number;
  }>;
}

// ============================================================================
// Block Diagram Generation Prompt
// ============================================================================

/**
 * Generate a prompt for creating a high-level block diagram
 */
export function generateBlockDiagramPrompt(
  requirements: ProjectRequirements,
  projectType: ProjectType
): LLMMessage[] {
  const systemPrompt = `You are an expert electrical engineer specializing in embedded systems and PCB design.
Your task is to analyze project requirements and create a comprehensive block diagram for a ${projectType.replace('_', ' ')} design.

You must respond with a valid JSON object containing the block diagram structure.

Guidelines:
1. Create functional blocks that logically group components
2. Define clear interfaces between blocks
3. Include all necessary power management blocks
4. Consider signal integrity and EMC from the start
5. Follow industry best practices for the project type
6. Include protection circuits where appropriate

For ${projectType.replace('_', ' ')} projects, pay special attention to:
${getProjectTypeGuidelines(projectType)}`;

  const userPrompt = `Create a block diagram for the following project:

**Project Name:** ${requirements.name}
**Description:** ${requirements.description}

**Power Requirements:**
${requirements.inputVoltage ? `- Input Voltage: ${formatVoltage(requirements.inputVoltage)}` : '- Input voltage not specified'}
${requirements.outputVoltages?.map(v => `- ${v.name}: ${v.voltage}V @ ${v.current}A`).join('\n') || '- Output voltages not specified'}
${requirements.maxCurrent ? `- Maximum Current: ${requirements.maxCurrent}A` : ''}

**MCU/Controller:**
${requirements.targetMcu || 'Not specified - recommend appropriate MCU'}

**Interfaces Required:**
${requirements.interfaces?.map(i => `- ${i.type.toUpperCase()}: ${i.count} instance(s)`).join('\n') || '- None specified'}

**Additional Features:**
${requirements.features?.map(f => `- ${f}`).join('\n') || '- None specified'}

**Design Constraints:**
${requirements.constraints?.maxComponents ? `- Maximum components: ${requirements.constraints.maxComponents}` : ''}
${requirements.constraints?.costTarget ? `- Cost target: $${requirements.constraints.costTarget}` : ''}
${requirements.constraints?.automotiveGrade ? '- Automotive grade components required (AEC-Q100/Q200)' : ''}
${requirements.constraints?.industrialGrade ? '- Industrial grade (-40C to +85C)' : ''}

Please provide a JSON response with this exact structure:
{
  "blocks": [
    {
      "id": "string - unique block identifier",
      "name": "string - human readable name",
      "type": "string - block type (power_input, voltage_regulator, mcu, protection, interface, etc)",
      "description": "string - what this block does",
      "inputs": ["list of input signal/power names"],
      "outputs": ["list of output signal/power names"],
      "components": ["list of main component types in this block"]
    }
  ],
  "connections": [
    {
      "from": { "block": "block_id", "port": "output_name" },
      "to": { "block": "block_id", "port": "input_name" },
      "signalType": "power|ground|analog|digital|differential"
    }
  ],
  "powerRails": [
    {
      "name": "string - rail name like VIN, 3V3, 5V",
      "voltage": number,
      "current": number,
      "source": "block_id that generates this rail"
    }
  ],
  "notes": ["any important design notes or recommendations"]
}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

/**
 * Validator for block diagram response
 */
export function validateBlockDiagramResponse(response: string): SchematicBlockDiagram | null {
  try {
    const parsed = JSON.parse(response);

    // Validate required fields
    if (!Array.isArray(parsed.blocks) || parsed.blocks.length === 0) {
      return null;
    }

    // Validate block structure
    for (const block of parsed.blocks) {
      if (!block.id || !block.name || !block.type || !block.description) {
        return null;
      }
      if (!Array.isArray(block.inputs) || !Array.isArray(block.outputs)) {
        return null;
      }
    }

    // Validate connections
    if (!Array.isArray(parsed.connections)) {
      return null;
    }

    // Validate power rails
    if (!Array.isArray(parsed.powerRails)) {
      return null;
    }

    return {
      blocks: parsed.blocks as SchematicBlock[],
      connections: parsed.connections,
      powerRails: parsed.powerRails,
      notes: parsed.notes || [],
    };
  } catch {
    return null;
  }
}

// ============================================================================
// Component Selection Prompt
// ============================================================================

/**
 * Generate a prompt for selecting components for a schematic block
 */
export function generateComponentSelectionPrompt(
  block: BlockRequirements,
  requirements: ProjectRequirements
): LLMMessage[] {
  const systemPrompt = `You are an expert electronics component engineer with deep knowledge of:
- Semiconductor manufacturers and their product lines
- Passive component specifications and trade-offs
- Component availability and lifecycle status
- EMC and thermal considerations
- Cost optimization strategies

Your task is to select specific components for a schematic block, providing real part numbers from major manufacturers.

Guidelines:
1. Select components that are currently in production and available
2. Prefer manufacturers with good automotive/industrial track records
3. Consider second-source availability
4. Balance cost vs performance vs reliability
5. Account for derating factors in the design
6. Include all supporting passive components

You must respond with a valid JSON array of component selections.`;

  const userPrompt = `Select components for the following schematic block:

**Block:** ${block.blockName} (${block.blockType})
**Description:** ${block.description}

**Input Signals:**
${block.inputSignals.map(s => `- ${s}`).join('\n')}

**Output Signals:**
${block.outputSignals.map(s => `- ${s}`).join('\n')}

**Power Rails Available:**
${block.powerRails.map(r => `- ${r}`).join('\n')}

**Project Context:**
- Target MCU: ${requirements.targetMcu || 'General purpose'}
- Input Voltage: ${formatVoltage(requirements.inputVoltage)}
${requirements.constraints?.automotiveGrade ? '- AUTOMOTIVE GRADE REQUIRED (AEC-Q100/Q200)' : ''}
${requirements.constraints?.industrialGrade ? '- Industrial temperature range required' : ''}
${block.constraints?.priceTarget ? `- Price target for block: $${block.constraints.priceTarget}` : ''}
${block.constraints?.preferredManufacturers?.length ? `- Preferred manufacturers: ${block.constraints.preferredManufacturers.join(', ')}` : ''}
${block.constraints?.avoidManufacturers?.length ? `- Avoid manufacturers: ${block.constraints.avoidManufacturers.join(', ')}` : ''}

Please provide a JSON array with this structure:
[
  {
    "reference": "string - reference designator like U1, R1, C1",
    "partNumber": "string - actual manufacturer part number",
    "manufacturer": "string - manufacturer name",
    "value": "string - component value or description",
    "footprint": "string - package/footprint like 0603, SOIC-8, QFN-48",
    "description": "string - brief description of function in circuit",
    "alternatives": ["array of alternative part numbers"],
    "reasoning": "string - why this component was selected"
  }
]

Include ALL components needed for the block including:
- Main ICs
- Decoupling capacitors (100nF + bulk)
- Bias resistors
- Filter components
- Protection devices (TVS, fuses)
- Connectors if applicable`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

/**
 * Validator for component selection response
 */
export function validateComponentSelectionResponse(response: string): ComponentSelection[] | null {
  try {
    const parsed = JSON.parse(response);

    if (!Array.isArray(parsed)) {
      return null;
    }

    const components: ComponentSelection[] = [];
    for (const item of parsed) {
      if (!item.reference || !item.partNumber || !item.manufacturer || !item.footprint) {
        continue; // Skip invalid entries but don't fail entirely
      }

      components.push({
        reference: item.reference,
        partNumber: item.partNumber,
        manufacturer: item.manufacturer,
        value: item.value || '',
        footprint: item.footprint,
        description: item.description || '',
        alternatives: Array.isArray(item.alternatives) ? item.alternatives : [],
        reasoning: item.reasoning || '',
      });
    }

    return components.length > 0 ? components : null;
  } catch {
    return null;
  }
}

// ============================================================================
// Netlist Generation Prompt
// ============================================================================

/**
 * Generate a prompt for creating the netlist from schematic data
 */
export function generateNetlistPrompt(schematic: SchematicForValidation): LLMMessage[] {
  const systemPrompt = `You are an expert electronics design engineer creating a netlist from schematic component data.

Your task is to:
1. Verify all connections are correct and complete
2. Assign appropriate net names following conventions
3. Identify net classes (power, signal, differential pairs)
4. Flag any potential connectivity issues
5. Apply proper naming conventions for power and ground nets

Net naming conventions:
- Power: VIN, 3V3, 5V, VBAT, etc.
- Ground: GND, AGND, DGND, PGND
- Signals: Descriptive names like UART1_TX, SPI_MOSI, LED_CTRL
- Differential: Use _P/_N suffixes (USB_D_P, USB_D_N)

You must respond with a valid JSON object containing the netlist.`;

  const userPrompt = `Generate a complete netlist for this schematic:

**Schematic:** ${schematic.name}

**Components:**
${schematic.components.map(c => `- ${c.reference}: ${c.value}${c.partNumber ? ` (${c.partNumber})` : ''}\n  Pins: ${c.pins.join(', ')}`).join('\n')}

**Existing Net Connections:**
${schematic.nets.map(n => `- ${n.name}: ${n.connections.join(', ')}`).join('\n')}

**Power Rails:**
${schematic.powerRails.map(p => `- ${p.name}: ${p.voltage}V${p.maxCurrent ? ` @ ${p.maxCurrent}A max` : ''}`).join('\n')}

Please provide a JSON response with this structure:
{
  "nets": [
    {
      "net": "string - net name",
      "connections": [
        { "component": "reference", "pin": "pin name or number" }
      ],
      "netClass": "power|ground|signal|high_speed|differential",
      "properties": {
        "voltage": number (if power net),
        "maxCurrent": number (if applicable),
        "impedance": number (if controlled impedance),
        "differentialPair": "string - paired net name" (if differential)
      }
    }
  ],
  "warnings": ["any connectivity warnings or concerns"],
  "recommendations": ["suggestions for improving the design"]
}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

/**
 * Validator for netlist response
 */
export function validateNetlistResponse(response: string): NetlistNode[] | null {
  try {
    const parsed = JSON.parse(response);

    if (!parsed.nets || !Array.isArray(parsed.nets)) {
      return null;
    }

    const nets: NetlistNode[] = [];
    for (const net of parsed.nets) {
      if (!net.net || !Array.isArray(net.connections)) {
        continue;
      }

      nets.push({
        net: net.net,
        connections: net.connections.map((c: { component: string; pin: string }) => ({
          component: c.component,
          pin: c.pin,
        })),
        netClass: net.netClass || 'signal',
        properties: net.properties || {},
      });
    }

    return nets.length > 0 ? nets : null;
  } catch {
    return null;
  }
}

// ============================================================================
// Schematic Validation Prompt
// ============================================================================

/**
 * Generate a prompt for validating a complete schematic
 */
export function validateSchematicPrompt(schematic: SchematicForValidation): LLMMessage[] {
  const systemPrompt = `You are a senior electronics design engineer performing a thorough design review.

Your expertise covers:
1. ERC (Electrical Rules Check) - proper connections, no floating pins
2. Power integrity - decoupling, voltage levels, current capacity
3. Signal integrity - termination, EMC, crosstalk
4. Component selection - appropriate values, ratings, availability
5. Best practices - industry standards, common pitfalls
6. Safety - protection circuits, failure modes

Review the schematic thoroughly and provide actionable feedback.
Be specific about issues and provide concrete solutions.

You must respond with a valid JSON object containing your review.`;

  const userPrompt = `Review this schematic for errors, warnings, and improvements:

**Schematic:** ${schematic.name}

**Components (${schematic.components.length} total):**
${schematic.components.map(c => `- ${c.reference}: ${c.value}`).join('\n')}

**Nets (${schematic.nets.length} total):**
${schematic.nets.map(n => `- ${n.name} (${n.type || 'signal'}): ${n.connections.length} connections`).join('\n')}

**Power Rails:**
${schematic.powerRails.map(p => `- ${p.name}: ${p.voltage}V @ ${p.maxCurrent || '?'}A`).join('\n')}

Please provide a comprehensive review in this JSON format:
{
  "score": number (0-100),
  "passed": boolean (true if score >= 80 and no critical issues),
  "summary": "string - one paragraph summary of the design quality",
  "criticalIssues": [
    {
      "code": "string - error code like ERC001",
      "message": "string - description of the issue",
      "location": "string - component reference or net name",
      "fix": "string - how to fix it"
    }
  ],
  "warnings": [
    {
      "code": "string - warning code",
      "message": "string - description",
      "location": "string - where",
      "suggestion": "string - recommendation"
    }
  ],
  "improvements": [
    {
      "category": "power|signal|thermal|emc|cost|reliability",
      "suggestion": "string - what could be improved",
      "benefit": "string - why it matters",
      "priority": "high|medium|low"
    }
  ],
  "checklist": {
    "powerDecoupling": boolean,
    "groundIntegrity": boolean,
    "noFloatingPins": boolean,
    "correctVoltages": boolean,
    "currentRatings": boolean,
    "protectionCircuits": boolean,
    "terminationPresent": boolean
  }
}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

// ============================================================================
// Helper Functions
// ============================================================================

function formatVoltage(voltage: number | { min: number; max: number } | undefined): string {
  if (!voltage) return 'Not specified';
  if (typeof voltage === 'number') return `${voltage}V`;
  return `${voltage.min}V - ${voltage.max}V`;
}

function getProjectTypeGuidelines(projectType: ProjectType): string {
  const guidelines: Record<ProjectType, string> = {
    motor_controller: `- Gate driver isolation and timing
- Current sensing and protection
- PWM filtering and deadtime
- Heat dissipation from power stage
- EMC filtering on motor connections
- Brake and regen circuitry`,

    power_supply: `- Input protection (TVS, MOV, fuse)
- Soft-start circuitry
- Feedback loop stability
- Output ripple and noise
- Overcurrent and overvoltage protection
- Thermal management`,

    sensor_node: `- Sensor signal conditioning
- ADC reference quality
- Low-power operation modes
- Environmental protection
- Calibration provisions
- Noise immunity`,

    communication_gateway: `- Multiple protocol level shifting
- Galvanic isolation where needed
- ESD protection on external interfaces
- Protocol-specific termination
- Status indicators
- Reset and watchdog circuits`,

    data_logger: `- Non-volatile storage interface
- RTC backup power
- Timestamp accuracy
- Data integrity mechanisms
- Power management for battery life
- Tamper detection if needed`,

    battery_management: `- Cell balancing circuits
- Coulomb counting accuracy
- Temperature monitoring per cell
- Protection (OVP, UVP, OCP, OTP)
- Charging control
- State of charge estimation`,

    led_driver: `- Constant current control
- PWM dimming interface
- Thermal foldback
- Open/short LED protection
- EMC filtering
- Color consistency`,

    audio_amplifier: `- Input coupling and filtering
- Bias stability
- Output protection
- Thermal management
- EMC compliance
- Power supply rejection`,

    rf_transceiver: `- Impedance matching networks
- RF filtering
- Shield partitioning
- Crystal/oscillator stability
- Power supply filtering
- Antenna interface`,

    general: `- Proper power sequencing
- Reset circuit reliability
- Debug interface accessibility
- Status indication
- Protection circuits
- Manufacturing testability`,
  };

  return guidelines[projectType] || guidelines.general;
}

// ============================================================================
// Exports
// ============================================================================

export default {
  generateBlockDiagramPrompt,
  validateBlockDiagramResponse,
  generateComponentSelectionPrompt,
  validateComponentSelectionResponse,
  generateNetlistPrompt,
  validateNetlistResponse,
  validateSchematicPrompt,
};
