#!/usr/bin/env python3
"""
Schematic PDF Exporter

Exports KiCad schematics to professional PDF documentation.
Works without KiCad installation by parsing the schematic files directly.

Features:
- Multi-sheet support
- Component table generation
- Net list documentation
- Wire/component metrics
- Title block extraction
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether, Flowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle, Group
    from reportlab.graphics import renderPDF
except ImportError:
    print("ERROR: reportlab not installed. Run: pip install reportlab", file=sys.stderr)
    sys.exit(1)

try:
    import sexpdata
except ImportError:
    print("ERROR: sexpdata not installed. Run: pip install sexpdata", file=sys.stderr)
    sys.exit(1)


@dataclass
class Position:
    """2D position."""
    x: float
    y: float


@dataclass
class Component:
    """Schematic component."""
    reference: str
    value: str
    symbol: str
    footprint: str
    position: Position
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class Wire:
    """Wire segment."""
    start: Position
    end: Position


@dataclass
class Label:
    """Net label."""
    text: str
    position: Position
    label_type: str = 'local'


@dataclass
class SheetData:
    """Parsed sheet data."""
    name: str
    title: str = ''
    revision: str = ''
    date: str = ''
    company: str = ''
    components: List[Component] = field(default_factory=list)
    wires: List[Wire] = field(default_factory=list)
    labels: List[Label] = field(default_factory=list)


class SchematicBlockDiagram(Flowable):
    """
    Generates a visual block diagram representation of the schematic.

    This creates a simplified schematic view showing:
    - Component blocks grouped by type
    - Net connections between blocks
    - Power rails
    """

    BRAND_PRIMARY = colors.HexColor("#4faeca")
    BRAND_ACCENT = colors.HexColor("#2ecc71")
    BRAND_DARK = colors.HexColor("#1a1a2e")
    BRAND_CARD = colors.HexColor("#16213e")

    # Component type colors
    TYPE_COLORS = {
        'Q': colors.HexColor('#e74c3c'),   # MOSFETs - red
        'U': colors.HexColor('#3498db'),   # ICs - blue
        'C': colors.HexColor('#27ae60'),   # Capacitors - green
        'R': colors.HexColor('#f39c12'),   # Resistors - orange
        'D': colors.HexColor('#9b59b6'),   # Diodes - purple
        'L': colors.HexColor('#1abc9c'),   # Inductors - teal
        'J': colors.HexColor('#34495e'),   # Connectors - gray
        '#': colors.HexColor('#95a5a6'),   # Power symbols - light gray
    }

    def __init__(self, sheet: 'SheetData', width: float = 6.5*inch, height: float = 4*inch):
        Flowable.__init__(self)
        self.sheet = sheet
        self.width = width
        self.height = height

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        return (self.width, self.height)

    def draw(self):
        """Draw the schematic block diagram."""
        # Background
        self.canv.setFillColor(colors.HexColor('#f7fafc'))
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)

        # Border
        self.canv.setStrokeColor(colors.HexColor('#cbd5e0'))
        self.canv.setLineWidth(1)
        self.canv.rect(0, 0, self.width, self.height, fill=0, stroke=1)

        # Title bar
        self.canv.setFillColor(self.BRAND_DARK)
        self.canv.rect(0, self.height - 25, self.width, 25, fill=1, stroke=0)

        self.canv.setFillColor(colors.white)
        self.canv.setFont('Helvetica-Bold', 10)
        title = self.sheet.name.replace('_', ' ')
        self.canv.drawString(10, self.height - 17, f"Block Diagram: {title}")

        # Group components by type
        comp_groups = self._group_components()

        if not comp_groups:
            self.canv.setFillColor(colors.HexColor('#718096'))
            self.canv.setFont('Helvetica', 10)
            self.canv.drawCentredString(self.width/2, self.height/2, "No components to display")
            return

        # Calculate layout
        available_height = self.height - 40
        available_width = self.width - 20

        # Draw component groups as blocks
        num_groups = len(comp_groups)
        cols = min(4, num_groups)
        rows = (num_groups + cols - 1) // cols

        block_width = (available_width - (cols-1)*10) / cols
        block_height = min(60, (available_height - (rows-1)*10) / rows)

        x_start = 10
        y_start = self.height - 35 - block_height

        for idx, (comp_type, comps) in enumerate(comp_groups.items()):
            col = idx % cols
            row = idx // cols

            x = x_start + col * (block_width + 10)
            y = y_start - row * (block_height + 10)

            self._draw_component_block(x, y, block_width, block_height, comp_type, comps)

        # Draw net labels as connection indicators
        self._draw_net_indicators(x_start, 20, available_width)

    def _group_components(self) -> Dict[str, List[Component]]:
        """Group components by type prefix."""
        groups: Dict[str, List[Component]] = {}

        for comp in self.sheet.components:
            prefix = ''
            for char in comp.reference:
                if char.isalpha() or char == '#':
                    prefix += char
                else:
                    break

            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(comp)

        # Sort groups by count (largest first)
        return dict(sorted(groups.items(), key=lambda x: -len(x[1])))

    def _draw_component_block(self, x: float, y: float, width: float, height: float,
                              comp_type: str, components: List[Component]):
        """Draw a component group block."""
        # Background
        color = self.TYPE_COLORS.get(comp_type, colors.HexColor('#7f8c8d'))
        self.canv.setFillColor(color)
        self.canv.roundRect(x, y, width, height, 5, fill=1, stroke=0)

        # Slightly darker border
        self.canv.setStrokeColor(colors.HexColor('#2d3748'))
        self.canv.setLineWidth(0.5)
        self.canv.roundRect(x, y, width, height, 5, fill=0, stroke=1)

        # Type label
        self.canv.setFillColor(colors.white)
        self.canv.setFont('Helvetica-Bold', 12)

        type_names = {
            'Q': 'MOSFETs',
            'U': 'ICs',
            'C': 'Capacitors',
            'R': 'Resistors',
            'D': 'Diodes',
            'L': 'Inductors',
            'J': 'Connectors',
            '#': 'Power',
        }
        type_name = type_names.get(comp_type, comp_type)
        self.canv.drawCentredString(x + width/2, y + height - 18, type_name)

        # Count
        self.canv.setFont('Helvetica', 20)
        self.canv.drawCentredString(x + width/2, y + height/2 - 10, str(len(components)))

        # Reference range
        self.canv.setFont('Helvetica', 8)
        refs = sorted([c.reference for c in components])
        if len(refs) <= 3:
            ref_text = ', '.join(refs)
        else:
            ref_text = f"{refs[0]} ... {refs[-1]}"

        if len(ref_text) > 20:
            ref_text = ref_text[:17] + '...'
        self.canv.drawCentredString(x + width/2, y + 8, ref_text)

    def _draw_net_indicators(self, x: float, y: float, width: float):
        """Draw net label indicators at the bottom."""
        labels = self.sheet.labels
        if not labels:
            return

        # Group labels by type
        global_labels = [l for l in labels if l.label_type in ('global', 'hierarchical')]
        power_labels = [l for l in labels if 'VCC' in l.text or 'GND' in l.text or 'V' in l.text]

        # Draw power rails indicator
        self.canv.setFillColor(self.BRAND_DARK)
        self.canv.setFont('Helvetica-Bold', 8)
        self.canv.drawString(x, y + 5, f"Nets: {len(labels)}")

        # Show power nets
        if power_labels:
            power_text = ', '.join(sorted(set(l.text for l in power_labels[:5])))
            if len(power_labels) > 5:
                power_text += f' (+{len(power_labels) - 5} more)'
            self.canv.setFont('Helvetica', 8)
            self.canv.setFillColor(colors.HexColor('#e74c3c'))
            self.canv.drawString(x + 60, y + 5, f"Power: {power_text}")


class SchematicParser:
    """Parser for KiCad schematic files."""

    def __init__(self, file_path: str):
        self.path = file_path
        self.sexp_data = None

    def parse(self) -> SheetData:
        """Parse the schematic file."""
        with open(self.path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.sexp_data = sexpdata.loads(content)

        sheet = SheetData(name=Path(self.path).stem)

        # Parse title block
        self._parse_title_block(sheet)

        # Parse components (placed symbols only)
        self._parse_components(sheet)

        # Parse wires
        self._parse_wires(sheet)

        # Parse labels
        self._parse_labels(sheet)

        return sheet

    def _find_node(self, node: Any, name: str) -> Optional[Any]:
        """Find a node by name."""
        if isinstance(node, list):
            for item in node:
                if isinstance(item, list) and len(item) > 0:
                    if hasattr(item[0], 'value') and item[0].value() == name:
                        return item
                    result = self._find_node(item, name)
                    if result:
                        return result
        return None

    def _find_all_nodes(self, node: Any, name: str) -> List[Any]:
        """Find all nodes with given name."""
        results = []
        if isinstance(node, list):
            for item in node:
                if isinstance(item, list) and len(item) > 0:
                    if hasattr(item[0], 'value') and item[0].value() == name:
                        results.append(item)
                    results.extend(self._find_all_nodes(item, name))
        return results

    def _get_value(self, node: List, key: str) -> Optional[str]:
        """Get value for a key."""
        for item in node:
            if isinstance(item, list) and len(item) >= 2:
                if hasattr(item[0], 'value') and item[0].value() == key:
                    return str(item[1])
        return None

    def _parse_title_block(self, sheet: SheetData):
        """Parse title block."""
        title_block = self._find_node(self.sexp_data, 'title_block')
        if title_block:
            sheet.title = self._get_value(title_block, 'title') or ''
            sheet.revision = self._get_value(title_block, 'rev') or ''
            sheet.date = self._get_value(title_block, 'date') or ''
            sheet.company = self._get_value(title_block, 'company') or ''

    def _parse_components(self, sheet: SheetData):
        """Parse placed components."""
        all_symbols = self._find_all_nodes(self.sexp_data, 'symbol')

        for sym in all_symbols:
            # Check if this is a placed symbol (has lib_id child)
            has_lib_id = any(
                isinstance(item, list) and len(item) > 0 and
                hasattr(item[0], 'value') and item[0].value() == 'lib_id'
                for item in sym[1:] if isinstance(item, list)
            )

            if not has_lib_id:
                continue

            # Get lib_id
            lib_id = ''
            for item in sym[1:]:
                if isinstance(item, list) and len(item) > 1:
                    if hasattr(item[0], 'value') and item[0].value() == 'lib_id':
                        lib_id = str(item[1])
                        break

            # Get position
            at_node = self._find_node(sym, 'at')
            position = Position(0, 0)
            if at_node and len(at_node) >= 3:
                position = Position(float(at_node[1]), float(at_node[2]))

            # Get properties
            properties = {}
            reference = ''
            value = ''
            footprint = ''

            property_nodes = self._find_all_nodes(sym, 'property')
            for prop in property_nodes:
                if len(prop) >= 3:
                    prop_name = str(prop[1])
                    prop_value = str(prop[2])
                    properties[prop_name] = prop_value

                    if prop_name == 'Reference':
                        reference = prop_value
                    elif prop_name == 'Value':
                        value = prop_value
                    elif prop_name == 'Footprint':
                        footprint = prop_value

            if reference:
                sheet.components.append(Component(
                    reference=reference,
                    value=value,
                    symbol=lib_id,
                    footprint=footprint,
                    position=position,
                    properties=properties
                ))

    def _parse_wires(self, sheet: SheetData):
        """Parse wires."""
        wire_nodes = self._find_all_nodes(self.sexp_data, 'wire')

        for wire_node in wire_nodes:
            pts_node = self._find_node(wire_node, 'pts')
            if not pts_node:
                continue

            xy_nodes = self._find_all_nodes(pts_node, 'xy')
            if len(xy_nodes) >= 2:
                start = Position(float(xy_nodes[0][1]), float(xy_nodes[0][2]))
                end = Position(float(xy_nodes[1][1]), float(xy_nodes[1][2]))
                sheet.wires.append(Wire(start=start, end=end))

    def _parse_labels(self, sheet: SheetData):
        """Parse labels."""
        for label_type in ['label', 'global_label', 'hierarchical_label', 'power_label']:
            label_nodes = self._find_all_nodes(self.sexp_data, label_type)

            for label_node in label_nodes:
                if len(label_node) < 2:
                    continue

                text = str(label_node[1])

                at_node = self._find_node(label_node, 'at')
                position = Position(0, 0)
                if at_node and len(at_node) >= 3:
                    position = Position(float(at_node[1]), float(at_node[2]))

                sheet.labels.append(Label(
                    text=text,
                    position=position,
                    label_type=label_type.replace('_label', '')
                ))


class PDFExporter:
    """Generates professional PDF documentation from schematic data."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        self.elements = []

    def _setup_styles(self):
        """Setup custom styles."""
        # Modify existing Title style instead of adding new
        self.styles['Title'].fontSize = 24
        self.styles['Title'].spaceAfter = 30
        self.styles['Title'].alignment = TA_CENTER
        self.styles['Title'].textColor = colors.HexColor('#1a365d')

        self.styles.add(ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a5568')
        ))

        self.styles.add(ParagraphStyle(
            'SheetTitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceBefore=20,
            spaceAfter=15,
            textColor=colors.HexColor('#2d3748')
        ))

        self.styles.add(ParagraphStyle(
            'SectionTitle',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#4a5568')
        ))

    def export(self, sheets: List[SheetData], project_name: str = 'Schematic'):
        """Export sheets to PDF."""
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Title page
        self._add_title_page(project_name, sheets)

        # Table of contents
        self._add_table_of_contents(sheets)

        # Sheet documentation
        for i, sheet in enumerate(sheets, 1):
            self.elements.append(PageBreak())
            self._add_sheet_documentation(sheet, i)

        # Summary page
        self.elements.append(PageBreak())
        self._add_summary(sheets)

        # Build PDF
        doc.build(self.elements)
        print(f"PDF exported: {self.output_path}")

    def _add_title_page(self, project_name: str, sheets: List[SheetData]):
        """Add title page."""
        self.elements.append(Spacer(1, 2*inch))

        self.elements.append(Paragraph(
            project_name.replace('_', ' ').replace('-', ' ').title(),
            self.styles['Title']
        ))

        self.elements.append(Paragraph(
            "Schematic Documentation",
            self.styles['CustomSubtitle']
        ))

        self.elements.append(Spacer(1, inch))

        # Project info table
        total_components = sum(len(s.components) for s in sheets)
        total_wires = sum(len(s.wires) for s in sheets)
        total_nets = sum(len(s.labels) for s in sheets)

        info_data = [
            ['Project', project_name],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M')],
            ['Sheets', str(len(sheets))],
            ['Total Components', str(total_components)],
            ['Total Wires', str(total_wires)],
            ['Total Net Labels', str(total_nets)],
        ]

        if sheets and sheets[0].company:
            info_data.insert(1, ['Company', sheets[0].company])
        if sheets and sheets[0].revision:
            info_data.insert(2, ['Revision', sheets[0].revision])

        info_table = Table(info_data, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))

        self.elements.append(info_table)

    def _add_table_of_contents(self, sheets: List[SheetData]):
        """Add table of contents."""
        self.elements.append(PageBreak())
        self.elements.append(Paragraph("Table of Contents", self.styles['SheetTitle']))
        self.elements.append(Spacer(1, 0.25*inch))

        toc_data = [['#', 'Sheet Name', 'Components', 'Wires', 'Ratio']]

        for i, sheet in enumerate(sheets, 1):
            comp_count = len(sheet.components)
            wire_count = len(sheet.wires)
            ratio = wire_count / comp_count if comp_count > 0 else 0

            toc_data.append([
                str(i),
                sheet.name.replace('_', ' '),
                str(comp_count),
                str(wire_count),
                f'{ratio:.2f}'
            ])

        toc_table = Table(toc_data, colWidths=[0.5*inch, 3*inch, 1*inch, 1*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))

        self.elements.append(toc_table)

    def _add_sheet_documentation(self, sheet: SheetData, sheet_num: int):
        """Add documentation for a single sheet."""
        # Sheet header
        self.elements.append(Paragraph(
            f"Sheet {sheet_num}: {sheet.name.replace('_', ' ')}",
            self.styles['SheetTitle']
        ))

        if sheet.title:
            self.elements.append(Paragraph(sheet.title, self.styles['Normal']))

        self.elements.append(Spacer(1, 0.25*inch))

        # Metrics
        comp_count = len(sheet.components)
        wire_count = len(sheet.wires)
        label_count = len(sheet.labels)
        ratio = wire_count / comp_count if comp_count > 0 else 0

        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Components', str(comp_count), '-'],
            ['Wires', str(wire_count), '-'],
            ['Net Labels', str(label_count), '-'],
            ['Wire/Component Ratio', f'{ratio:.2f}', 'PASS' if ratio >= 1.2 else 'LOW'],
        ]

        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1*inch])
        metrics_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (2, 4), (2, 4),
             colors.HexColor('#c6f6d5') if ratio >= 1.2 else colors.HexColor('#fed7d7')),
        ]))

        self.elements.append(metrics_table)
        self.elements.append(Spacer(1, 0.25*inch))

        # Block diagram visualization
        if sheet.components:
            self.elements.append(Paragraph("Block Diagram", self.styles['SectionTitle']))
            block_diagram = SchematicBlockDiagram(sheet, width=6.5*inch, height=3.5*inch)
            self.elements.append(block_diagram)
            self.elements.append(Spacer(1, 0.25*inch))

        # Component table
        if sheet.components:
            self._add_component_table(sheet.components)

        # Net labels
        if sheet.labels:
            self._add_label_table(sheet.labels)

    def _add_component_table(self, components: List[Component]):
        """Add component bill of materials table."""
        self.elements.append(Paragraph("Components", self.styles['SectionTitle']))

        # Sort by reference
        sorted_comps = sorted(components, key=lambda c: (
            re.match(r'^([A-Z]+)', c.reference).group(1) if re.match(r'^([A-Z]+)', c.reference) else '',
            int(re.search(r'\d+', c.reference).group()) if re.search(r'\d+', c.reference) else 0
        ))

        # Group by type
        comp_data = [['Reference', 'Value', 'Footprint']]

        for comp in sorted_comps:
            footprint = comp.footprint.split(':')[-1] if comp.footprint else ''
            comp_data.append([
                comp.reference,
                comp.value[:30] if comp.value else '',
                footprint[:25] if footprint else ''
            ])

        # Split into chunks if too many components
        chunk_size = 40
        for i in range(0, len(comp_data), chunk_size):
            chunk = comp_data[i:i+chunk_size]
            if i > 0:
                chunk.insert(0, comp_data[0])  # Add header

            comp_table = Table(chunk, colWidths=[1.2*inch, 2.5*inch, 2.5*inch])
            comp_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
            ]))

            self.elements.append(comp_table)

            if i + chunk_size < len(comp_data):
                self.elements.append(Spacer(1, 0.1*inch))

    def _add_label_table(self, labels: List[Label]):
        """Add net labels table."""
        self.elements.append(Spacer(1, 0.25*inch))
        self.elements.append(Paragraph("Net Labels", self.styles['SectionTitle']))

        # Group by type
        power_labels = [l for l in labels if l.label_type == 'power']
        global_labels = [l for l in labels if l.label_type == 'global']
        local_labels = [l for l in labels if l.label_type == 'local']

        label_data = [['Net Name', 'Type']]

        for label in sorted(set(l.text for l in power_labels)):
            label_data.append([label, 'Power'])
        for label in sorted(set(l.text for l in global_labels)):
            label_data.append([label, 'Global'])
        for label in sorted(set(l.text for l in local_labels)):
            label_data.append([label, 'Local'])

        if len(label_data) > 1:
            label_table = Table(label_data, colWidths=[4*inch, 1.5*inch])
            label_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
            ]))

            self.elements.append(label_table)

    def _add_summary(self, sheets: List[SheetData]):
        """Add summary page."""
        self.elements.append(Paragraph("Validation Summary", self.styles['SheetTitle']))
        self.elements.append(Spacer(1, 0.25*inch))

        # Overall metrics
        total_components = sum(len(s.components) for s in sheets)
        total_wires = sum(len(s.wires) for s in sheets)
        overall_ratio = total_wires / total_components if total_components > 0 else 0

        all_pass = all(
            len(s.wires) / len(s.components) >= 1.2 if len(s.components) > 0 else True
            for s in sheets
        )

        summary_data = [
            ['Metric', 'Value'],
            ['Total Sheets', str(len(sheets))],
            ['Total Components', str(total_components)],
            ['Total Wires', str(total_wires)],
            ['Overall Wire/Component Ratio', f'{overall_ratio:.2f}'],
            ['Validation Status', 'PASS' if all_pass else 'NEEDS REVIEW'],
        ]

        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (1, 5), (1, 5),
             colors.HexColor('#c6f6d5') if all_pass else colors.HexColor('#feebc8')),
        ]))

        self.elements.append(summary_table)

        # Per-sheet validation
        self.elements.append(Spacer(1, 0.5*inch))
        self.elements.append(Paragraph("Per-Sheet Validation", self.styles['SectionTitle']))

        validation_data = [['Sheet', 'Components', 'Wires', 'Ratio', 'Status']]

        for sheet in sheets:
            comp_count = len(sheet.components)
            wire_count = len(sheet.wires)
            ratio = wire_count / comp_count if comp_count > 0 else 0
            status = 'PASS' if ratio >= 1.2 else 'LOW RATIO'

            validation_data.append([
                sheet.name.replace('_', ' ')[:25],
                str(comp_count),
                str(wire_count),
                f'{ratio:.2f}',
                status
            ])

        validation_table = Table(validation_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        validation_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))

        # Color status cells
        for i, sheet in enumerate(sheets, 1):
            comp_count = len(sheet.components)
            wire_count = len(sheet.wires)
            ratio = wire_count / comp_count if comp_count > 0 else 0
            color = colors.HexColor('#c6f6d5') if ratio >= 1.2 else colors.HexColor('#fed7d7')
            validation_table.setStyle(TableStyle([
                ('BACKGROUND', (4, i), (4, i), color),
            ]))

        self.elements.append(validation_table)

        # Footer
        self.elements.append(Spacer(1, inch))
        self.elements.append(Paragraph(
            f"Generated by Adverant Schematic-Gen Plugin | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ParagraphStyle('Footer', parent=self.styles['Normal'],
                          fontSize=9, textColor=colors.gray, alignment=TA_CENTER)
        ))


def main():
    parser = argparse.ArgumentParser(
        description='Export KiCad schematics to PDF documentation'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input schematic file(s), comma-separated or glob pattern'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output PDF file path'
    )
    parser.add_argument(
        '--project-name', '-n',
        type=str,
        default='Schematic',
        help='Project name for title page'
    )

    args = parser.parse_args()

    # Parse input files
    input_files = []
    for pattern in args.input.split(','):
        pattern = pattern.strip()
        if '*' in pattern or '?' in pattern:
            input_files.extend(sorted(Path('.').glob(pattern)))
        elif os.path.exists(pattern):
            input_files.append(Path(pattern))
        else:
            print(f"WARNING: File not found: {pattern}", file=sys.stderr)

    if not input_files:
        print("ERROR: No input files found", file=sys.stderr)
        sys.exit(1)

    # Parse schematics
    sheets = []
    for file_path in input_files:
        try:
            parser_obj = SchematicParser(str(file_path))
            sheet = parser_obj.parse()
            sheets.append(sheet)
            print(f"Parsed: {file_path.name} ({len(sheet.components)} components, {len(sheet.wires)} wires)")
        except Exception as e:
            print(f"ERROR parsing {file_path}: {e}", file=sys.stderr)

    if not sheets:
        print("ERROR: No sheets parsed successfully", file=sys.stderr)
        sys.exit(1)

    # Export to PDF
    exporter = PDFExporter(args.output)
    exporter.export(sheets, args.project_name)


if __name__ == '__main__':
    main()
