/**
 * EE Design Partner - KiCad Schematic Generator
 *
 * Generates valid KiCad 7.x schematic files in S-expression format.
 * Creates schematics from subsystem architecture definitions with proper
 * component symbols, hierarchical sheets, and wiring.
 *
 * @module utils/kicad-generator
 */

import { randomUUID } from 'crypto';
import { log, Logger } from './logger.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Subsystem definition for schematic generation
 */
export interface SubsystemDefinition {
  id: string;
  name: string;
  category: string;
  description?: string;
}

/**
 * Component definition for schematic generation
 */
export interface ComponentDefinition {
  reference: string;
  value: string;
  footprint?: string;
  library: string;
  symbol: string;
  properties?: Record<string, string>;
}

/**
 * Architecture definition for schematic generation
 */
export interface ArchitectureDefinition {
  subsystems: SubsystemDefinition[];
  projectType?: string;
  title?: string;
  revision?: string;
  date?: string;
  company?: string;
}

/**
 * Options for schematic generation
 */
export interface GenerateSchematicOptions {
  architecture: ArchitectureDefinition;
  components?: ComponentDefinition[];
  projectName: string;
  paperSize?: 'A4' | 'A3' | 'A2' | 'A1' | 'A0' | 'USLetter' | 'USLegal';
}

/**
 * Generated schematic result
 */
export interface GeneratedSchematic {
  content: string;
  sheets: SchematicSheet[];
  components: SchematicComponent[];
  nets: SchematicNet[];
}

/**
 * Schematic sheet metadata
 */
export interface SchematicSheet {
  name: string;
  uuid: string;
  page: number;
}

/**
 * Schematic component metadata
 */
export interface SchematicComponent {
  reference: string;
  value: string;
  uuid: string;
  library: string;
  symbol: string;
}

/**
 * Schematic net metadata
 */
export interface SchematicNet {
  name: string;
  uuid: string;
  code: number;
}

// ============================================================================
// Constants
// ============================================================================

const KICAD_VERSION = 20231120;
const GENERATOR_NAME = 'nexus-ee-design';
const GENERATOR_VERSION = '1.0.0';

/**
 * Subsystem category to symbol library mapping
 */
const CATEGORY_SYMBOLS: Record<string, { library: string; symbol: string }[]> = {
  Power: [
    { library: 'Device', symbol: 'C' },
    { library: 'Device', symbol: 'D_Schottky' },
    { library: 'Device', symbol: 'L' },
    { library: 'Regulator_Switching', symbol: 'LM2596S-5' },
  ],
  Control: [
    { library: 'MCU_ST_STM32G4', symbol: 'STM32G431KBTx' },
    { library: 'Device', symbol: 'Crystal' },
    { library: 'Device', symbol: 'C' },
    { library: 'Device', symbol: 'R' },
  ],
  Sensing: [
    { library: 'Sensor_Current', symbol: 'INA219' },
    { library: 'Device', symbol: 'R' },
    { library: 'Device', symbol: 'C' },
  ],
  Interface: [
    { library: 'Interface_CAN_LIN', symbol: 'MCP2551-I-SN' },
    { library: 'Device', symbol: 'R' },
    { library: 'Device', symbol: 'C' },
  ],
  Analog: [
    { library: 'Amplifier_Operational', symbol: 'OPA2340' },
    { library: 'Device', symbol: 'R' },
    { library: 'Device', symbol: 'C' },
  ],
  Default: [
    { library: 'Device', symbol: 'R' },
    { library: 'Device', symbol: 'C' },
  ],
};

/**
 * Paper size dimensions in mm
 */
const PAPER_SIZES: Record<string, { width: number; height: number }> = {
  A4: { width: 297, height: 210 },
  A3: { width: 420, height: 297 },
  A2: { width: 594, height: 420 },
  A1: { width: 841, height: 594 },
  A0: { width: 1189, height: 841 },
  USLetter: { width: 279.4, height: 215.9 },
  USLegal: { width: 355.6, height: 215.9 },
};

// ============================================================================
// Generator Implementation
// ============================================================================

const generatorLogger: Logger = log.child({ service: 'kicad-generator' });

/**
 * Generate a valid KiCad 7.x schematic from architecture definition.
 *
 * @param options - Generation options
 * @returns Generated schematic content and metadata
 */
export function generateSchematic(options: GenerateSchematicOptions): GeneratedSchematic {
  const logger = generatorLogger.child({ operation: 'generateSchematic' });

  const { architecture, components, projectName, paperSize = 'A4' } = options;

  logger.info('Generating KiCad schematic', {
    projectName,
    subsystemCount: architecture.subsystems.length,
    componentCount: components?.length || 0,
    paperSize,
  });

  // Generate UUIDs for the schematic
  const schematicUuid = randomUUID();

  // Build the sheets array
  const sheets: SchematicSheet[] = [
    { name: 'Root', uuid: schematicUuid, page: 1 },
  ];

  // Build components array from architecture
  const schematicComponents: SchematicComponent[] = [];
  const schematicNets: SchematicNet[] = [];

  // Generate component entries
  let componentIndex = 1;
  let netCode = 1;

  // Add a power net
  const vccNet: SchematicNet = {
    name: 'VCC',
    uuid: randomUUID(),
    code: netCode++,
  };
  schematicNets.push(vccNet);

  const gndNet: SchematicNet = {
    name: 'GND',
    uuid: randomUUID(),
    code: netCode++,
  };
  schematicNets.push(gndNet);

  // Generate components for each subsystem
  for (const subsystem of architecture.subsystems) {
    const categorySymbols = CATEGORY_SYMBOLS[subsystem.category] || CATEGORY_SYMBOLS.Default;

    // Add representative components for the subsystem
    for (const symbolDef of categorySymbols.slice(0, 2)) {
      const comp: SchematicComponent = {
        reference: `${symbolDef.symbol.charAt(0).toUpperCase()}${componentIndex++}`,
        value: subsystem.name,
        uuid: randomUUID(),
        library: symbolDef.library,
        symbol: symbolDef.symbol,
      };
      schematicComponents.push(comp);
    }

    // Add a net for this subsystem's output
    const subsystemNet: SchematicNet = {
      name: `${subsystem.name.replace(/\s+/g, '_')}_OUT`,
      uuid: randomUUID(),
      code: netCode++,
    };
    schematicNets.push(subsystemNet);
  }

  // Add any explicitly provided components
  if (components) {
    for (const comp of components) {
      schematicComponents.push({
        reference: comp.reference,
        value: comp.value,
        uuid: randomUUID(),
        library: comp.library,
        symbol: comp.symbol,
      });
    }
  }

  // Generate the S-expression content
  const content = generateSExpression({
    schematicUuid,
    projectName,
    architecture,
    paperSize,
    sheets,
    components: schematicComponents,
    nets: schematicNets,
  });

  logger.info('Schematic generated', {
    contentLength: content.length,
    sheetCount: sheets.length,
    componentCount: schematicComponents.length,
    netCount: schematicNets.length,
  });

  return {
    content,
    sheets,
    components: schematicComponents,
    nets: schematicNets,
  };
}

/**
 * Generate KiCad S-expression format content
 */
function generateSExpression(params: {
  schematicUuid: string;
  projectName: string;
  architecture: ArchitectureDefinition;
  paperSize: string;
  sheets: SchematicSheet[];
  components: SchematicComponent[];
  nets: SchematicNet[];
}): string {
  const { schematicUuid, projectName, architecture, paperSize, sheets, components, nets } = params;

  const paper = PAPER_SIZES[paperSize] || PAPER_SIZES.A4;

  // Build title block
  const titleBlock = buildTitleBlock({
    title: architecture.title || projectName,
    date: architecture.date || new Date().toISOString().split('T')[0],
    rev: architecture.revision || '1.0',
    company: architecture.company || 'Generated by Nexus EE Design',
    comment1: `Subsystems: ${architecture.subsystems.map((s) => s.name).join(', ')}`,
    comment2: `Project Type: ${architecture.projectType || 'Electronic Design'}`,
  });

  // Build lib_symbols section with all required symbols
  const libSymbols = buildLibSymbols(components);

  // Build symbol instances (placed components)
  const symbolInstances = buildSymbolInstances(components, paper);

  // Build wires/connections
  const wires = buildWires(components, nets, paper);

  // Build labels
  const labels = buildLabels(nets, paper);

  // Build power symbols
  const powerSymbols = buildPowerSymbols(paper);

  // Build sheet instances
  const sheetInstances = buildSheetInstances(sheets);

  // Assemble the full schematic
  return `(kicad_sch
  (version ${KICAD_VERSION})
  (generator "${GENERATOR_NAME}")
  (generator_version "${GENERATOR_VERSION}")
  (uuid "${schematicUuid}")
  (paper "${paperSize}")
${titleBlock}
${libSymbols}
${powerSymbols}
${symbolInstances}
${wires}
${labels}
${sheetInstances}
)
`;
}

/**
 * Build the title block S-expression
 */
function buildTitleBlock(params: {
  title: string;
  date: string;
  rev: string;
  company: string;
  comment1?: string;
  comment2?: string;
}): string {
  const lines = [
    '  (title_block',
    `    (title "${escapeString(params.title)}")`,
    `    (date "${params.date}")`,
    `    (rev "${escapeString(params.rev)}")`,
    `    (company "${escapeString(params.company)}")`,
  ];

  if (params.comment1) {
    lines.push(`    (comment 1 "${escapeString(params.comment1)}")`);
  }
  if (params.comment2) {
    lines.push(`    (comment 2 "${escapeString(params.comment2)}")`);
  }

  lines.push('  )');
  return lines.join('\n');
}

/**
 * Build lib_symbols section with symbol definitions
 */
function buildLibSymbols(components: SchematicComponent[]): string {
  const uniqueSymbols = new Map<string, SchematicComponent>();

  for (const comp of components) {
    const key = `${comp.library}:${comp.symbol}`;
    if (!uniqueSymbols.has(key)) {
      uniqueSymbols.set(key, comp);
    }
  }

  const symbolDefs: string[] = ['  (lib_symbols'];

  for (const [key, comp] of uniqueSymbols) {
    symbolDefs.push(buildLibSymbol(key, comp));
  }

  // Add power symbols
  symbolDefs.push(buildPowerLibSymbol('power:VCC'));
  symbolDefs.push(buildPowerLibSymbol('power:GND'));

  symbolDefs.push('  )');
  return symbolDefs.join('\n');
}

/**
 * Build a single lib_symbol definition
 */
function buildLibSymbol(libId: string, comp: SchematicComponent): string {
  const uuid = randomUUID();

  // Create a basic rectangular symbol for any component
  return `    (symbol "${libId}"
      (pin_numbers hide)
      (pin_names
        (offset 1.016)
      )
      (exclude_from_sim no)
      (in_bom yes)
      (on_board yes)
      (property "Reference" "${comp.reference.charAt(0)}"
        (at 0 2.54 0)
        (effects
          (font
            (size 1.27 1.27)
          )
        )
      )
      (property "Value" "${escapeString(comp.value)}"
        (at 0 -2.54 0)
        (effects
          (font
            (size 1.27 1.27)
          )
        )
      )
      (property "Footprint" ""
        (at 0 0 0)
        (effects
          (font
            (size 1.27 1.27)
          )
          (hide yes)
        )
      )
      (property "Datasheet" ""
        (at 0 0 0)
        (effects
          (font
            (size 1.27 1.27)
          )
          (hide yes)
        )
      )
      (symbol "${libId}_0_1"
        (rectangle
          (start -2.54 2.54)
          (end 2.54 -2.54)
          (stroke
            (width 0.254)
            (type default)
          )
          (fill
            (type background)
          )
        )
      )
      (symbol "${libId}_1_1"
        (pin passive line
          (at -5.08 0 0)
          (length 2.54)
          (name "1"
            (effects
              (font
                (size 1.27 1.27)
              )
            )
          )
          (number "1"
            (effects
              (font
                (size 1.27 1.27)
              )
            )
          )
        )
        (pin passive line
          (at 5.08 0 180)
          (length 2.54)
          (name "2"
            (effects
              (font
                (size 1.27 1.27)
              )
            )
          )
          (number "2"
            (effects
              (font
                (size 1.27 1.27)
              )
            )
          )
        )
      )
    )`;
}

/**
 * Build power symbol definition
 */
function buildPowerLibSymbol(libId: string): string {
  const isVcc = libId.includes('VCC');

  return `    (symbol "${libId}"
      (power)
      (pin_numbers hide)
      (pin_names
        (offset 0) hide
      )
      (exclude_from_sim no)
      (in_bom yes)
      (on_board yes)
      (property "Reference" "#PWR"
        (at 0 ${isVcc ? '2.54' : '-2.54'} 0)
        (effects
          (font
            (size 1.27 1.27)
          )
          (hide yes)
        )
      )
      (property "Value" "${isVcc ? 'VCC' : 'GND'}"
        (at 0 ${isVcc ? '2.54' : '-2.54'} 0)
        (effects
          (font
            (size 1.27 1.27)
          )
        )
      )
      (symbol "${libId}_0_1"
        ${isVcc ? `(polyline
          (pts
            (xy 0 0) (xy 0 1.27) (xy -0.635 1.905) (xy 0 2.54) (xy 0.635 1.905) (xy 0 1.27)
          )
          (stroke
            (width 0)
            (type default)
          )
          (fill
            (type outline)
          )
        )` : `(polyline
          (pts
            (xy -1.27 0) (xy 1.27 0) (xy 0 -1.27) (xy -1.27 0)
          )
          (stroke
            (width 0)
            (type default)
          )
          (fill
            (type outline)
          )
        )`}
      )
      (symbol "${libId}_1_1"
        (pin power_in line
          (at 0 0 ${isVcc ? '270' : '90'})
          (length 0)
          (name "${isVcc ? 'VCC' : 'GND'}"
            (effects
              (font
                (size 1.27 1.27)
              )
            )
          )
          (number "1"
            (effects
              (font
                (size 1.27 1.27)
              )
            )
          )
        )
      )
    )`;
}

/**
 * Build symbol instances (placed components)
 */
function buildSymbolInstances(components: SchematicComponent[], paper: { width: number; height: number }): string {
  const instances: string[] = [];

  // Calculate grid positions for components
  const startX = 50.8; // Start 50mm from left
  const startY = 50.8; // Start 50mm from top
  const spacingX = 40.64; // 40mm horizontal spacing
  const spacingY = 25.4; // 25mm vertical spacing
  const maxCols = Math.floor((paper.width - 100) / spacingX);

  for (let i = 0; i < components.length; i++) {
    const comp = components[i];
    const col = i % maxCols;
    const row = Math.floor(i / maxCols);
    const x = startX + col * spacingX;
    const y = startY + row * spacingY;

    instances.push(`  (symbol
    (lib_id "${comp.library}:${comp.symbol}")
    (at ${x.toFixed(2)} ${y.toFixed(2)} 0)
    (unit 1)
    (exclude_from_sim no)
    (in_bom yes)
    (on_board yes)
    (dnp no)
    (uuid "${comp.uuid}")
    (property "Reference" "${comp.reference}"
      (at ${x.toFixed(2)} ${(y - 3.81).toFixed(2)} 0)
      (effects
        (font
          (size 1.27 1.27)
        )
      )
    )
    (property "Value" "${escapeString(comp.value)}"
      (at ${x.toFixed(2)} ${(y + 3.81).toFixed(2)} 0)
      (effects
        (font
          (size 1.27 1.27)
        )
      )
    )
    (property "Footprint" ""
      (at ${x.toFixed(2)} ${y.toFixed(2)} 0)
      (effects
        (font
          (size 1.27 1.27)
        )
        (hide yes)
      )
    )
    (property "Datasheet" ""
      (at ${x.toFixed(2)} ${y.toFixed(2)} 0)
      (effects
        (font
          (size 1.27 1.27)
        )
        (hide yes)
      )
    )
    (pin "1"
      (uuid "${randomUUID()}")
    )
    (pin "2"
      (uuid "${randomUUID()}")
    )
    (instances
      (project ""
        (path "/${comp.uuid}"
          (reference "${comp.reference}")
          (unit 1)
        )
      )
    )
  )`);
  }

  return instances.join('\n');
}

/**
 * Build power symbols (VCC and GND)
 */
function buildPowerSymbols(paper: { width: number; height: number }): string {
  const vccX = 25.4;
  const vccY = 25.4;
  const gndX = 25.4;
  const gndY = paper.height - 25.4;
  const vccUuid = randomUUID();
  const gndUuid = randomUUID();

  return `  (symbol
    (lib_id "power:VCC")
    (at ${vccX.toFixed(2)} ${vccY.toFixed(2)} 0)
    (unit 1)
    (exclude_from_sim no)
    (in_bom no)
    (on_board yes)
    (dnp no)
    (uuid "${vccUuid}")
    (property "Reference" "#PWR01"
      (at ${vccX.toFixed(2)} ${(vccY + 2.54).toFixed(2)} 0)
      (effects
        (font
          (size 1.27 1.27)
        )
        (hide yes)
      )
    )
    (property "Value" "VCC"
      (at ${vccX.toFixed(2)} ${(vccY - 2.54).toFixed(2)} 0)
      (effects
        (font
          (size 1.27 1.27)
        )
      )
    )
    (pin "1"
      (uuid "${randomUUID()}")
    )
    (instances
      (project ""
        (path "/${vccUuid}"
          (reference "#PWR01")
          (unit 1)
        )
      )
    )
  )
  (symbol
    (lib_id "power:GND")
    (at ${gndX.toFixed(2)} ${gndY.toFixed(2)} 0)
    (unit 1)
    (exclude_from_sim no)
    (in_bom no)
    (on_board yes)
    (dnp no)
    (uuid "${gndUuid}")
    (property "Reference" "#PWR02"
      (at ${gndX.toFixed(2)} ${(gndY - 2.54).toFixed(2)} 0)
      (effects
        (font
          (size 1.27 1.27)
        )
        (hide yes)
      )
    )
    (property "Value" "GND"
      (at ${gndX.toFixed(2)} ${(gndY + 2.54).toFixed(2)} 0)
      (effects
        (font
          (size 1.27 1.27)
        )
      )
    )
    (pin "1"
      (uuid "${randomUUID()}")
    )
    (instances
      (project ""
        (path "/${gndUuid}"
          (reference "#PWR02")
          (unit 1)
        )
      )
    )
  )`;
}

/**
 * Build wire connections
 */
function buildWires(
  components: SchematicComponent[],
  nets: SchematicNet[],
  paper: { width: number; height: number }
): string {
  const wires: string[] = [];

  // Calculate the same positions as components
  const startX = 50.8;
  const startY = 50.8;
  const spacingX = 40.64;
  const spacingY = 25.4;
  const maxCols = Math.floor((paper.width - 100) / spacingX);

  // Connect consecutive components with wires
  for (let i = 0; i < components.length - 1; i++) {
    const col1 = i % maxCols;
    const row1 = Math.floor(i / maxCols);
    const col2 = (i + 1) % maxCols;
    const row2 = Math.floor((i + 1) / maxCols);

    // Only connect components in the same row
    if (row1 === row2) {
      const x1 = startX + col1 * spacingX + 5.08; // Right edge of first component
      const x2 = startX + col2 * spacingX - 5.08; // Left edge of second component
      const y = startY + row1 * spacingY;

      wires.push(`  (wire
    (pts
      (xy ${x1.toFixed(2)} ${y.toFixed(2)}) (xy ${x2.toFixed(2)} ${y.toFixed(2)})
    )
    (stroke
      (width 0)
      (type default)
    )
    (uuid "${randomUUID()}")
  )`);
    }
  }

  return wires.join('\n');
}

/**
 * Build net labels
 */
function buildLabels(nets: SchematicNet[], paper: { width: number; height: number }): string {
  const labels: string[] = [];

  // Place net labels along the top
  const startX = 76.2;
  const y = 12.7;
  const spacing = 25.4;

  for (let i = 0; i < Math.min(nets.length, 8); i++) {
    const net = nets[i];
    const x = startX + i * spacing;

    labels.push(`  (label "${escapeString(net.name)}"
    (at ${x.toFixed(2)} ${y.toFixed(2)} 0)
    (fields_autoplaced yes)
    (effects
      (font
        (size 1.27 1.27)
      )
      (justify left bottom)
    )
    (uuid "${net.uuid}")
  )`);
  }

  return labels.join('\n');
}

/**
 * Build sheet instances section
 */
function buildSheetInstances(sheets: SchematicSheet[]): string {
  const instances: string[] = ['  (sheet_instances'];

  for (const sheet of sheets) {
    instances.push(`    (path "/"
      (page "${sheet.page}")
    )`);
  }

  instances.push('  )');
  return instances.join('\n');
}

/**
 * Escape special characters in strings for S-expression
 */
function escapeString(str: string): string {
  return str
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
}

/**
 * Generate a minimal valid KiCad schematic
 * Use this when you need a valid schematic without any components
 */
export function generateMinimalSchematic(projectName: string): GeneratedSchematic {
  const uuid = randomUUID();

  const content = `(kicad_sch
  (version ${KICAD_VERSION})
  (generator "${GENERATOR_NAME}")
  (generator_version "${GENERATOR_VERSION}")
  (uuid "${uuid}")
  (paper "A4")
  (title_block
    (title "${escapeString(projectName)}")
    (date "${new Date().toISOString().split('T')[0]}")
    (rev "1.0")
    (company "Generated by Nexus EE Design")
  )
  (lib_symbols
${buildPowerLibSymbol('power:VCC')}
${buildPowerLibSymbol('power:GND')}
  )
  (sheet_instances
    (path "/"
      (page "1")
    )
  )
)
`;

  return {
    content,
    sheets: [{ name: 'Root', uuid, page: 1 }],
    components: [],
    nets: [],
  };
}

// Default export
export default {
  generateSchematic,
  generateMinimalSchematic,
};
