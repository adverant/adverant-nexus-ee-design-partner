/**
 * Default Compliance Checklist (MAPO v3.0)
 *
 * 51 static compliance checks across 6 standards.
 * Each check is executed by schematic-compliance-validator.ts service.
 */

import {
  ChecklistItemDefinition,
  ComplianceStandard,
  CheckCategory,
  ViolationSeverity
} from '../types/schematic-quality';

export const DEFAULT_CHECKLIST: ChecklistItemDefinition[] = [
  // ============================================
  // NASA-STD-8739.4: Workmanship Standards (10 checks)
  // ============================================
  {
    id: 'nasa-power-01',
    name: 'Power Decoupling Capacitors',
    description: 'Verify all ICs have decoupling capacitors on power pins',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.POWER,
    severity: ViolationSeverity.HIGH,
    rationale: 'Decoupling capacitors filter noise and prevent voltage drops on IC power pins, critical for stable operation.',
    remediation: 'Add 0.1µF ceramic capacitor between VCC and GND pins of each IC, placed as close as possible to the IC.',
    autoFixable: false,
    references: ['NASA-STD-8739.4 §3.2.1']
  },
  {
    id: 'nasa-power-02',
    name: 'Power and Ground Symbols',
    description: 'All power/ground nets use proper symbols (not wire labels)',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.SYMBOLS,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Power symbols provide visual clarity and prevent accidental connections.',
    remediation: 'Replace wire labels like "VCC" or "GND" with official power/ground symbols from symbol library.',
    autoFixable: true,
    references: ['NASA-STD-8739.4 §4.1.2']
  },
  {
    id: 'nasa-conn-01',
    name: 'No Unconnected Pins',
    description: 'All IC pins are either connected or marked as "No Connect"',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.HIGH,
    rationale: 'Unconnected pins can cause undefined behavior or ESD damage.',
    remediation: 'Connect unused pins per datasheet recommendations or add "No Connect" flag.',
    autoFixable: false,
    references: ['NASA-STD-8739.4 §3.3.4']
  },
  {
    id: 'nasa-conn-02',
    name: 'Net Naming Conventions',
    description: 'All nets follow naming convention (uppercase, descriptive, no spaces)',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.LOW,
    rationale: 'Consistent naming improves readability and reduces errors.',
    remediation: 'Rename nets to follow convention: VCC_3V3, SPI_MOSI, UART_TX, etc.',
    autoFixable: true,
    references: ['NASA-STD-8739.4 §4.2.1']
  },
  {
    id: 'nasa-doc-01',
    name: 'Required Component Properties',
    description: 'All components have Reference, Value, Footprint, and Description',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Complete properties enable automated BOM generation and manufacturing.',
    remediation: 'Add missing properties to components: Reference (R1, C5), Value (10K, 0.1µF), Footprint (0805), Description.',
    autoFixable: false,
    references: ['NASA-STD-8739.4 §5.1.3']
  },
  {
    id: 'nasa-doc-02',
    name: 'Schematic Title Block',
    description: 'Title block contains project name, revision, date, and author',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.LOW,
    rationale: 'Title block provides essential metadata for document control.',
    remediation: 'Fill in title block fields with project details.',
    autoFixable: false,
    references: ['NASA-STD-8739.4 §5.2.1']
  },
  {
    id: 'nasa-place-01',
    name: 'Component Grid Alignment',
    description: 'All components aligned to 50 mil (1.27mm) grid',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.LOW,
    rationale: 'Grid alignment improves visual cleanliness and wire routing.',
    remediation: 'Move components to snap to grid (Edit → Align to Grid).',
    autoFixable: true,
    references: ['NASA-STD-8739.4 §4.3.2']
  },
  {
    id: 'nasa-place-02',
    name: 'Signal Flow Direction',
    description: 'Signals flow left-to-right, power flows top-to-bottom',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Consistent signal flow follows industry standards and improves readability.',
    remediation: 'Rearrange components so inputs are on left, outputs on right, power at top.',
    autoFixable: false,
    references: ['NASA-STD-8739.4 §4.3.1']
  },
  {
    id: 'nasa-std-01',
    name: 'Symbol Library Standard',
    description: 'All symbols from approved KiCad libraries (no custom symbols)',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.SYMBOLS,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Standard symbols ensure consistency and manufacturability.',
    remediation: 'Replace custom symbols with official KiCad library symbols.',
    autoFixable: false,
    references: ['NASA-STD-8739.4 §3.1.1']
  },
  {
    id: 'nasa-std-02',
    name: 'Wire Junction Dots',
    description: 'All wire junctions have junction dots (not T-junctions)',
    standards: [ComplianceStandard.NASA_STD_8739_4],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.HIGH,
    rationale: 'Junction dots clarify intentional connections vs. crossovers.',
    remediation: 'Add junction dots at all 3-way and 4-way wire intersections.',
    autoFixable: true,
    references: ['NASA-STD-8739.4 §4.2.3']
  },

  // ============================================
  // MIL-STD-883: Microelectronics (8 checks)
  // ============================================
  {
    id: 'mil-power-01',
    name: 'Power Supply Filtering',
    description: 'Bulk capacitors (10µF-100µF) present on each power rail',
    standards: [ComplianceStandard.MIL_STD_883],
    category: CheckCategory.POWER,
    severity: ViolationSeverity.HIGH,
    rationale: 'Bulk capacitors provide local energy storage and low-frequency filtering.',
    remediation: 'Add bulk capacitor (10µF+ tantalum or ceramic) on each VCC net near power entry.',
    autoFixable: false,
    references: ['MIL-STD-883 Method 3015']
  },
  {
    id: 'mil-power-02',
    name: 'Separate Analog/Digital Grounds',
    description: 'Analog and digital grounds connected at single point only',
    standards: [ComplianceStandard.MIL_STD_883],
    category: CheckCategory.POWER,
    severity: ViolationSeverity.HIGH,
    rationale: 'Single-point ground connection prevents digital noise coupling to analog circuits.',
    remediation: 'Split AGND and DGND nets, connect only at power supply or via ferrite bead.',
    autoFixable: false,
    references: ['MIL-STD-883 Method 3012']
  },
  {
    id: 'mil-conn-01',
    name: 'ESD Protection on Inputs',
    description: 'All external inputs have ESD protection (TVS diodes or clamp circuits)',
    standards: [ComplianceStandard.MIL_STD_883],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.CRITICAL,
    rationale: 'ESD protection prevents IC damage from electrostatic discharge events.',
    remediation: 'Add TVS diodes or ESD clamp circuits on all connector pins and exposed I/O.',
    autoFixable: false,
    references: ['MIL-STD-883 Method 3015.7']
  },
  {
    id: 'mil-conn-02',
    name: 'Series Termination Resistors',
    description: 'High-speed signals (>10 MHz) have series termination resistors',
    standards: [ComplianceStandard.MIL_STD_883],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Termination resistors reduce reflections and improve signal integrity.',
    remediation: 'Add 22-100Ω series resistors on clock, SPI, and fast digital signals.',
    autoFixable: false,
    references: ['MIL-STD-883 Method 3012.4']
  },
  {
    id: 'mil-place-01',
    name: 'Thermal Relief on Power Pins',
    description: 'Power pins use thermal relief patterns (not solid fills)',
    standards: [ComplianceStandard.MIL_STD_883],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Thermal relief improves solderability by reducing heat sink effect.',
    remediation: 'Configure power pins with thermal relief spokes (4 connections, 0.4mm width).',
    autoFixable: true,
    references: ['MIL-STD-883 Method 2003']
  },
  {
    id: 'mil-std-01',
    name: 'Military-Grade Component Ratings',
    description: 'All components rated for -55°C to +125°C (or wider)',
    standards: [ComplianceStandard.MIL_STD_883],
    category: CheckCategory.STANDARDS,
    severity: ViolationSeverity.INFO,
    rationale: 'Wide temperature rating ensures operation in harsh environments.',
    remediation: 'Select mil-spec or automotive-grade components with extended temperature range.',
    autoFixable: false,
    references: ['MIL-STD-883 §3.1']
  },
  {
    id: 'mil-std-02',
    name: 'Derating Guidelines',
    description: 'Resistors/capacitors derated to 50% max voltage/power',
    standards: [ComplianceStandard.MIL_STD_883],
    category: CheckCategory.STANDARDS,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Derating improves reliability and lifetime.',
    remediation: 'Verify component ratings: 6V caps on 3.3V rail, 0.25W resistors for 0.1W dissipation.',
    autoFixable: false,
    references: ['MIL-STD-883 §4.2']
  },
  {
    id: 'mil-doc-01',
    name: 'Component Traceability',
    description: 'All ICs have manufacturer part number in Description field',
    standards: [ComplianceStandard.MIL_STD_883],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Full part numbers enable procurement and counterfeit prevention.',
    remediation: 'Add manufacturer and part number to Description: "Texas Instruments TPS54331".',
    autoFixable: false,
    references: ['MIL-STD-883 §5.1.2']
  },

  // ============================================
  // IPC-2221: PCB Design Standards (10 checks)
  // ============================================
  {
    id: 'ipc-power-01',
    name: 'Power Trace Width',
    description: 'Power traces sized for current load (≥20 mil for 1A)',
    standards: [ComplianceStandard.IPC_2221],
    category: CheckCategory.POWER,
    severity: ViolationSeverity.HIGH,
    rationale: 'Adequate trace width prevents excessive voltage drop and heating.',
    remediation: 'Widen power traces: 20 mil (0.5mm) for 1A, 40 mil (1mm) for 2A, 80 mil (2mm) for 5A.',
    autoFixable: true,
    references: ['IPC-2221 §6.2']
  },
  {
    id: 'ipc-power-02',
    name: 'Via Stitching on Ground Planes',
    description: 'Ground plane via stitching every 1-2 inches (25-50mm)',
    standards: [ComplianceStandard.IPC_2221],
    category: CheckCategory.POWER,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Via stitching reduces ground plane impedance and improves EMI performance.',
    remediation: 'Add ground vias around ICs and along board edges, spaced ≤50mm apart.',
    autoFixable: false,
    references: ['IPC-2221 §6.4.3']
  },
  {
    id: 'ipc-conn-01',
    name: 'Differential Pair Matching',
    description: 'Differential pairs (USB, LVDS, CAN) have ±5% length matching',
    standards: [ComplianceStandard.IPC_2221],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.HIGH,
    rationale: 'Length matching minimizes skew and maintains signal integrity.',
    remediation: 'Route differential pairs together, match lengths within 5% (meander if needed).',
    autoFixable: false,
    references: ['IPC-2221 §7.3.4']
  },
  {
    id: 'ipc-conn-02',
    name: 'Minimum Trace Spacing',
    description: 'Traces spaced ≥6 mil (0.15mm) for standard fabrication',
    standards: [ComplianceStandard.IPC_2221],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.CRITICAL,
    rationale: 'Adequate spacing prevents shorts and ensures manufacturability.',
    remediation: 'Increase trace spacing to ≥6 mil (0.15mm). Use wider spacing (10+ mil) for high voltage.',
    autoFixable: true,
    references: ['IPC-2221 §6.3.1']
  },
  {
    id: 'ipc-place-01',
    name: 'Component Clearance',
    description: 'Components spaced ≥50 mil (1.27mm) from board edges',
    standards: [ComplianceStandard.IPC_2221],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Edge clearance prevents damage during depaneling and handling.',
    remediation: 'Move components away from board edges, maintain ≥50 mil keepout zone.',
    autoFixable: true,
    references: ['IPC-2221 §9.1.3']
  },
  {
    id: 'ipc-place-02',
    name: 'Silkscreen Readability',
    description: 'Reference designators visible and not obscured by components',
    standards: [ComplianceStandard.IPC_2221],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.LOW,
    rationale: 'Visible reference designators aid assembly and debug.',
    remediation: 'Move silkscreen text outside component body outlines.',
    autoFixable: true,
    references: ['IPC-2221 §10.2.1']
  },
  {
    id: 'ipc-std-01',
    name: 'Annular Ring Size',
    description: 'Via annular rings ≥4 mil (0.1mm) for Class 2 boards',
    standards: [ComplianceStandard.IPC_2221],
    category: CheckCategory.STANDARDS,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Adequate annular rings ensure reliable plating and connection.',
    remediation: 'Increase via pad size or decrease drill size to achieve ≥4 mil annular ring.',
    autoFixable: true,
    references: ['IPC-2221 §8.2.2']
  },
  {
    id: 'ipc-std-02',
    name: 'Solder Mask Expansion',
    description: 'Solder mask expands 4 mil (0.1mm) beyond pad edges',
    standards: [ComplianceStandard.IPC_2221],
    category: CheckCategory.STANDARDS,
    severity: ViolationSeverity.LOW,
    rationale: 'Mask expansion prevents solder bridging between pads.',
    remediation: 'Configure solder mask expansion to 4 mil in board stackup settings.',
    autoFixable: true,
    references: ['IPC-2221 §9.1.5']
  },
  {
    id: 'ipc-doc-01',
    name: 'Footprint Consistency',
    description: 'All footprints match IPC-7351 land pattern standards',
    standards: [ComplianceStandard.IPC_2221, ComplianceStandard.IPC_7351],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Standard footprints ensure assembly compatibility.',
    remediation: 'Replace custom footprints with IPC-7351 compliant patterns from KiCad library.',
    autoFixable: false,
    references: ['IPC-2221 §10.1', 'IPC-7351 §3.2']
  },
  {
    id: 'ipc-doc-02',
    name: 'Layer Stack Documentation',
    description: 'Layer stackup documented in schematic notes or separate file',
    standards: [ComplianceStandard.IPC_2221],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.LOW,
    rationale: 'Stackup documentation aids fabrication and impedance control.',
    remediation: 'Add layer stackup table to schematic or export as fabrication note.',
    autoFixable: false,
    references: ['IPC-2221 §2.2.1']
  },

  // ============================================
  // IEC 61000: EMC Standards (8 checks)
  // ============================================
  {
    id: 'iec-power-01',
    name: 'Common Mode Choke on Power Input',
    description: 'AC/DC power inputs have common mode choke for EMI filtering',
    standards: [ComplianceStandard.IEC_61000],
    category: CheckCategory.POWER,
    severity: ViolationSeverity.HIGH,
    rationale: 'Common mode chokes reduce conducted emissions per IEC 61000-4-6.',
    remediation: 'Add common mode choke (ferrite bead or dedicated inductor) on AC/DC input lines.',
    autoFixable: false,
    references: ['IEC 61000-4-6 §7.2']
  },
  {
    id: 'iec-power-02',
    name: 'Pi Filter on Switching Regulators',
    description: 'Switching regulators have LC or CLC output filters',
    standards: [ComplianceStandard.IEC_61000],
    category: CheckCategory.POWER,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Output filtering reduces switching noise and EMI.',
    remediation: 'Add inductor-capacitor filter after switching regulator output.',
    autoFixable: false,
    references: ['IEC 61000-4-4 §6.3']
  },
  {
    id: 'iec-conn-01',
    name: 'Shield Grounding on Cables',
    description: 'Shielded cable shields connected to chassis ground at one end',
    standards: [ComplianceStandard.IEC_61000],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.HIGH,
    rationale: 'Proper shield grounding reduces radiated and conducted emissions.',
    remediation: 'Connect cable shield to chassis ground at connector, not circuit ground.',
    autoFixable: false,
    references: ['IEC 61000-5-2 §8.4']
  },
  {
    id: 'iec-conn-02',
    name: 'Ferrite Beads on I/O Lines',
    description: 'External I/O lines have ferrite beads for high-frequency filtering',
    standards: [ComplianceStandard.IEC_61000],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Ferrite beads suppress high-frequency noise on cables.',
    remediation: 'Add ferrite beads (600Ω @ 100 MHz) on signal lines near connectors.',
    autoFixable: false,
    references: ['IEC 61000-4-3 §7.1']
  },
  {
    id: 'iec-place-01',
    name: 'Clock Trace Routing',
    description: 'Clock traces <3 inches (75mm) and routed away from board edges',
    standards: [ComplianceStandard.IEC_61000],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.HIGH,
    rationale: 'Short clock traces reduce radiated emissions per IEC 61000-4-3.',
    remediation: 'Shorten clock traces and route on inner layers away from edges.',
    autoFixable: false,
    references: ['IEC 61000-4-3 §6.2']
  },
  {
    id: 'iec-place-02',
    name: 'Ground Plane Under Oscillators',
    description: 'Crystal oscillators placed over solid ground plane',
    standards: [ComplianceStandard.IEC_61000],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.HIGH,
    rationale: 'Ground plane provides return path and reduces EMI.',
    remediation: 'Ensure no plane splits under oscillator, route traces on inner layers.',
    autoFixable: false,
    references: ['IEC 61000-4-6 §8.3']
  },
  {
    id: 'iec-std-01',
    name: 'EMC Test Point Access',
    description: 'Test points provided for EMC probe measurements',
    standards: [ComplianceStandard.IEC_61000],
    category: CheckCategory.STANDARDS,
    severity: ViolationSeverity.LOW,
    rationale: 'Test points aid EMC compliance testing and debug.',
    remediation: 'Add test points on power rails, ground, and critical signals.',
    autoFixable: false,
    references: ['IEC 61000-4-2 §9.1']
  },
  {
    id: 'iec-doc-01',
    name: 'EMC Compliance Statement',
    description: 'Schematic includes EMC design notes and intended standards',
    standards: [ComplianceStandard.IEC_61000],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.LOW,
    rationale: 'Documentation shows design intent for EMC compliance.',
    remediation: 'Add text note: "Designed for IEC 61000-4-2/3/4/6 compliance".',
    autoFixable: false,
    references: ['IEC 61000-1-2 §5.3']
  },

  // ============================================
  // IPC-7351: Land Pattern Standards (5 checks)
  // ============================================
  {
    id: 'ipc7351-std-01',
    name: 'Footprint Naming Convention',
    description: 'Footprints follow IPC-7351 naming: PKG_SIZE_PITCH_VARIANT',
    standards: [ComplianceStandard.IPC_7351],
    category: CheckCategory.STANDARDS,
    severity: ViolationSeverity.LOW,
    rationale: 'Standardized naming improves library management and sharing.',
    remediation: 'Rename footprints: SOIC-8_3.9x4.9mm_P1.27mm, QFN-32_5x5mm_P0.5mm.',
    autoFixable: true,
    references: ['IPC-7351 §2.1']
  },
  {
    id: 'ipc7351-std-02',
    name: 'Pad Size Tolerances',
    description: 'SMT pad dimensions within ±10% of IPC-7351 nominal',
    standards: [ComplianceStandard.IPC_7351],
    category: CheckCategory.STANDARDS,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Correct pad sizing ensures reliable solder joints.',
    remediation: 'Adjust pad dimensions to match IPC-7351 calculator or library values.',
    autoFixable: true,
    references: ['IPC-7351 §4.2.3']
  },
  {
    id: 'ipc7351-std-03',
    name: 'Courtyard Clearance',
    description: 'Component courtyards have ≥0.25mm clearance from adjacent components',
    standards: [ComplianceStandard.IPC_7351],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Courtyard clearance prevents assembly conflicts.',
    remediation: 'Increase component spacing to achieve ≥0.25mm courtyard gap.',
    autoFixable: true,
    references: ['IPC-7351 §5.1.2']
  },
  {
    id: 'ipc7351-std-04',
    name: 'Reference Designator Placement',
    description: 'Reference designators centered on component or in courtyard',
    standards: [ComplianceStandard.IPC_7351],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.LOW,
    rationale: 'Consistent ref des placement improves assembly efficiency.',
    remediation: 'Move reference designators to component center or standard position (top-left for ICs).',
    autoFixable: true,
    references: ['IPC-7351 §6.3.1']
  },
  {
    id: 'ipc7351-doc-01',
    name: '3D Model Association',
    description: 'All footprints have associated 3D STEP models',
    standards: [ComplianceStandard.IPC_7351],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.INFO,
    rationale: '3D models enable mechanical clearance checks and visualization.',
    remediation: 'Associate STEP models with footprints in library editor.',
    autoFixable: false,
    references: ['IPC-7351 §7.2']
  },

  // ============================================
  // Professional Best Practices (10 checks)
  // ============================================
  {
    id: 'bp-power-01',
    name: 'Reverse Polarity Protection',
    description: 'Power inputs have reverse polarity protection (diode or P-FET)',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.POWER,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Protection prevents damage from incorrect power connection.',
    remediation: 'Add series Schottky diode or P-channel MOSFET on power input.',
    autoFixable: false,
    references: ['Industry Best Practice']
  },
  {
    id: 'bp-power-02',
    name: 'Inrush Current Limiting',
    description: 'High-capacitance loads have inrush current limiting',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.POWER,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Limits current spike during power-up to protect power supply.',
    remediation: 'Add NTC thermistor or soft-start circuit on power input.',
    autoFixable: false,
    references: ['Industry Best Practice']
  },
  {
    id: 'bp-conn-01',
    name: 'Pull-up/Pull-down Resistors',
    description: 'All I2C/SPI lines and enable pins have pull resistors',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.HIGH,
    rationale: 'Pull resistors prevent floating inputs and ensure defined states.',
    remediation: 'Add 4.7K-10K pull-ups on I2C (SDA/SCL), pull-downs on active-low enables.',
    autoFixable: false,
    references: ['Industry Best Practice']
  },
  {
    id: 'bp-conn-02',
    name: 'Reset Circuit',
    description: 'Microcontroller has proper reset circuit (supervisor IC or RC circuit)',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.HIGH,
    rationale: 'Reset circuit ensures clean startup and prevents brown-out issues.',
    remediation: 'Add reset supervisor IC (e.g., TPS3840) or 10K/0.1µF RC reset circuit.',
    autoFixable: false,
    references: ['Industry Best Practice']
  },
  {
    id: 'bp-conn-03',
    name: 'Status LEDs',
    description: 'Power and status LEDs present for visual debug',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.CONNECTIVITY,
    severity: ViolationSeverity.LOW,
    rationale: 'LEDs provide immediate visual feedback during development and debug.',
    remediation: 'Add power LED on VCC rail and status LED on GPIO (with current-limiting resistor).',
    autoFixable: false,
    references: ['Industry Best Practice']
  },
  {
    id: 'bp-place-01',
    name: 'Test Point Placement',
    description: 'Test points on all power rails and critical signals',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Test points enable efficient debugging and production testing.',
    remediation: 'Add test points (1mm pad or through-hole) on VCC, GND, and critical signals.',
    autoFixable: false,
    references: ['Industry Best Practice']
  },
  {
    id: 'bp-place-02',
    name: 'Component Orientation Consistency',
    description: 'Similar components oriented consistently (e.g., all caps same direction)',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.PLACEMENT,
    severity: ViolationSeverity.LOW,
    rationale: 'Consistent orientation reduces assembly errors.',
    remediation: 'Rotate components to align similar parts in same direction.',
    autoFixable: true,
    references: ['Industry Best Practice']
  },
  {
    id: 'bp-doc-01',
    name: 'BOM Export Completeness',
    description: 'Schematic exports complete BOM with all required fields',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.MEDIUM,
    rationale: 'Complete BOM enables procurement and assembly.',
    remediation: 'Verify BOM export includes: Reference, Value, Footprint, Manufacturer, MPN, Quantity.',
    autoFixable: false,
    references: ['Industry Best Practice']
  },
  {
    id: 'bp-doc-02',
    name: 'Version Control Integration',
    description: 'Schematic includes git hash or version number',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.LOW,
    rationale: 'Version tracking enables reproducibility and debugging.',
    remediation: 'Add version number or git hash to schematic title block or text note.',
    autoFixable: false,
    references: ['Industry Best Practice']
  },
  {
    id: 'bp-std-01',
    name: 'Design Review Sign-off',
    description: 'Schematic marked as reviewed with reviewer name and date',
    standards: [ComplianceStandard.BEST_PRACTICES],
    category: CheckCategory.DOCUMENTATION,
    severity: ViolationSeverity.INFO,
    rationale: 'Formal review process improves quality and accountability.',
    remediation: 'Add "Reviewed by: [Name], Date: [YYYY-MM-DD]" to title block.',
    autoFixable: false,
    references: ['Industry Best Practice']
  }
];

/**
 * Get all checks for a specific standard
 */
export function getChecksByStandard(standard: ComplianceStandard): ChecklistItemDefinition[] {
  return DEFAULT_CHECKLIST.filter(check =>
    check.standards.includes(standard)
  );
}

/**
 * Get all checks for a specific category
 */
export function getChecksByCategory(category: CheckCategory): ChecklistItemDefinition[] {
  return DEFAULT_CHECKLIST.filter(check => check.category === category);
}

/**
 * Get all checks with specific severity
 */
export function getChecksBySeverity(severity: ViolationSeverity): ChecklistItemDefinition[] {
  return DEFAULT_CHECKLIST.filter(check => check.severity === severity);
}

/**
 * Get all auto-fixable checks
 */
export function getAutoFixableChecks(): ChecklistItemDefinition[] {
  return DEFAULT_CHECKLIST.filter(check => check.autoFixable);
}

/**
 * Get check by ID
 */
export function getCheckById(id: string): ChecklistItemDefinition | undefined {
  return DEFAULT_CHECKLIST.find(check => check.id === id);
}

/**
 * Get check counts by category
 */
export function getCheckCountsByCategory(): Record<CheckCategory, number> {
  const counts: Partial<Record<CheckCategory, number>> = {};
  for (const category of Object.values(CheckCategory)) {
    counts[category] = getChecksByCategory(category).length;
  }
  return counts as Record<CheckCategory, number>;
}

/**
 * Get check counts by standard
 */
export function getCheckCountsByStandard(): Record<ComplianceStandard, number> {
  const counts: Partial<Record<ComplianceStandard, number>> = {};
  for (const standard of Object.values(ComplianceStandard)) {
    counts[standard] = getChecksByStandard(standard).length;
  }
  return counts as Record<ComplianceStandard, number>;
}
