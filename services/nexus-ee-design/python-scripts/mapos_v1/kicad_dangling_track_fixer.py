#!/usr/bin/env python3
"""
KiCad Dangling Track Fixer

Handles track_dangling DRC violations by:
1. Identifying tracks that end without connecting to a pad, via, or another track
2. Either extending them to nearest connection point or removing them

Uses pcbnew API directly - no regex parsing.

Part of MAPOS Gaming AI Remediation - Phase 2
"""

import pcbnew
from pathlib import Path
import json
import argparse
import shutil
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional
import math


class DanglingTrackFixer:
    """Fix dangling track ends using pcbnew API."""

    # Tolerance for considering points as connected (in nm)
    CONNECTION_TOLERANCE = 1000  # 1 micron

    # Maximum distance to extend a track to reach a connection point (in nm)
    MAX_EXTENSION_DISTANCE = 5_000_000  # 5mm

    def __init__(self, pcb_path: str):
        """
        Initialize the fixer.

        Args:
            pcb_path: Path to .kicad_pcb file
        """
        self.pcb_path = Path(pcb_path)
        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_path}")

        self.board = pcbnew.LoadBoard(str(self.pcb_path))
        self._connection_points: Optional[Dict[str, Set[Tuple[int, int]]]] = None
        self._track_endpoints: Optional[Dict] = None

    def _get_connection_points(self) -> Dict[str, Set[Tuple[int, int]]]:
        """
        Get all valid connection points grouped by net name.

        Returns:
            Dict mapping net names to sets of (x, y) coordinates in nm
        """
        if self._connection_points is not None:
            return self._connection_points

        connection_points: Dict[str, Set[Tuple[int, int]]] = {}

        # Add pad positions
        for footprint in self.board.GetFootprints():
            for pad in footprint.Pads():
                net_name = pad.GetNetname()
                if not net_name:
                    continue

                pos = pad.GetPosition()
                point = (pos.x, pos.y)

                if net_name not in connection_points:
                    connection_points[net_name] = set()
                connection_points[net_name].add(point)

        # Add via positions
        for track in self.board.GetTracks():
            if track.GetClass() == 'PCB_VIA':
                net_name = track.GetNetname()
                if not net_name:
                    continue

                pos = track.GetPosition()
                point = (pos.x, pos.y)

                if net_name not in connection_points:
                    connection_points[net_name] = set()
                connection_points[net_name].add(point)

        self._connection_points = connection_points
        return connection_points

    def _get_track_endpoints(self) -> Dict:
        """
        Get all track endpoints with their connectivity info.

        Returns:
            Dict with track info including start/end points and connections
        """
        if self._track_endpoints is not None:
            return self._track_endpoints

        tracks_info = {}

        # First pass: collect all track endpoints
        for track in self.board.GetTracks():
            if track.GetClass() != 'PCB_TRACK':
                continue

            track_id = id(track)
            start = track.GetStart()
            end = track.GetEnd()

            tracks_info[track_id] = {
                'track': track,
                'start': (start.x, start.y),
                'end': (end.x, end.y),
                'net': track.GetNetname(),
                'layer': track.GetLayer(),
                'width': track.GetWidth(),
                'start_connected': False,
                'end_connected': False,
            }

        # Get connection points
        connection_points = self._get_connection_points()

        # Second pass: check connectivity
        for track_id, info in tracks_info.items():
            net = info['net']
            start = info['start']
            end = info['end']

            # Check if endpoints connect to pads/vias
            if net in connection_points:
                for conn_point in connection_points[net]:
                    if self._points_connected(start, conn_point):
                        info['start_connected'] = True
                    if self._points_connected(end, conn_point):
                        info['end_connected'] = True

            # Check if endpoints connect to other tracks
            for other_id, other_info in tracks_info.items():
                if other_id == track_id:
                    continue
                if other_info['net'] != net:
                    continue
                if other_info['layer'] != info['layer']:
                    continue

                # Check all endpoint combinations
                if self._points_connected(start, other_info['start']) or \
                   self._points_connected(start, other_info['end']):
                    info['start_connected'] = True

                if self._points_connected(end, other_info['start']) or \
                   self._points_connected(end, other_info['end']):
                    info['end_connected'] = True

        self._track_endpoints = tracks_info
        return tracks_info

    def _points_connected(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """Check if two points are close enough to be considered connected."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist = math.sqrt(dx * dx + dy * dy)
        return dist <= self.CONNECTION_TOLERANCE

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate distance between two points."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)

    def find_dangling_tracks(self) -> List[Dict]:
        """
        Find all tracks with dangling endpoints.

        Returns:
            List of dicts with dangling track info
        """
        dangling = []
        tracks_info = self._get_track_endpoints()

        for track_id, info in tracks_info.items():
            dangling_endpoints = []

            if not info['start_connected']:
                dangling_endpoints.append({
                    'endpoint': 'start',
                    'position': info['start'],
                })

            if not info['end_connected']:
                dangling_endpoints.append({
                    'endpoint': 'end',
                    'position': info['end'],
                })

            if dangling_endpoints:
                dangling.append({
                    'track_id': track_id,
                    'track': info['track'],
                    'net': info['net'],
                    'layer': info['layer'],
                    'dangling_endpoints': dangling_endpoints,
                })

        return dangling

    def remove_dangling_tracks(self, backup: bool = True, save: bool = True) -> Dict:
        """
        Remove tracks that have dangling endpoints.

        Note: This removes the entire track if ANY endpoint is dangling.
        For partial removal, use trim_dangling_tracks() instead.

        Args:
            backup: Create backup before modifying
            save: Save changes to file

        Returns:
            Dict with operation results
        """
        if backup:
            self._create_backup()

        dangling = self.find_dangling_tracks()
        removed = []

        for d in dangling:
            track = d['track']

            # Record info before removal
            removed.append({
                'net': d['net'],
                'layer': pcbnew.LayerName(d['layer']),
                'endpoints': [ep['position'] for ep in d['dangling_endpoints']],
            })

            # Remove the track
            self.board.Remove(track)

        if save and removed:
            pcbnew.SaveBoard(str(self.pcb_path), self.board)

        return {
            'success': True,
            'operation': 'remove_dangling_tracks',
            'tracks_analyzed': len(self._get_track_endpoints()),
            'dangling_found': len(dangling),
            'tracks_removed': len(removed),
            'removed_tracks': removed,
            'file_saved': save and len(removed) > 0,
            'backup_created': backup,
        }

    def extend_dangling_tracks(self, backup: bool = True, save: bool = True) -> Dict:
        """
        Try to extend dangling tracks to reach nearest valid connection point.

        Args:
            backup: Create backup before modifying
            save: Save changes to file

        Returns:
            Dict with operation results
        """
        if backup:
            self._create_backup()

        dangling = self.find_dangling_tracks()
        connection_points = self._get_connection_points()

        extended = []
        failed = []

        for d in dangling:
            track = d['track']
            net = d['net']
            layer = d['layer']

            if net not in connection_points:
                failed.append({
                    'net': net,
                    'reason': 'No connection points found for net',
                })
                continue

            for ep_info in d['dangling_endpoints']:
                endpoint = ep_info['position']

                # Find nearest connection point on same net
                nearest = None
                min_dist = float('inf')

                for conn_point in connection_points[net]:
                    dist = self._distance(endpoint, conn_point)

                    # Skip if already connected (within tolerance)
                    if dist <= self.CONNECTION_TOLERANCE:
                        continue

                    # Check distance constraint
                    if dist < min_dist and dist <= self.MAX_EXTENSION_DISTANCE:
                        min_dist = dist
                        nearest = conn_point

                if nearest:
                    # Create new track segment to bridge the gap
                    new_track = pcbnew.PCB_TRACK(self.board)
                    new_track.SetStart(pcbnew.VECTOR2I(endpoint[0], endpoint[1]))
                    new_track.SetEnd(pcbnew.VECTOR2I(nearest[0], nearest[1]))
                    new_track.SetWidth(track.GetWidth())
                    new_track.SetLayer(layer)
                    new_track.SetNet(track.GetNet())

                    self.board.Add(new_track)

                    extended.append({
                        'net': net,
                        'layer': pcbnew.LayerName(layer),
                        'from': endpoint,
                        'to': nearest,
                        'distance_nm': int(min_dist),
                        'distance_mm': round(min_dist / 1_000_000, 3),
                    })
                else:
                    failed.append({
                        'net': net,
                        'endpoint': endpoint,
                        'reason': f'No connection point within {self.MAX_EXTENSION_DISTANCE / 1_000_000}mm',
                    })

        if save and extended:
            pcbnew.SaveBoard(str(self.pcb_path), self.board)

        return {
            'success': True,
            'operation': 'extend_dangling_tracks',
            'dangling_found': sum(len(d['dangling_endpoints']) for d in dangling),
            'tracks_extended': len(extended),
            'extension_failed': len(failed),
            'extended_tracks': extended,
            'failures': failed,
            'file_saved': save and len(extended) > 0,
            'backup_created': backup,
        }

    def trim_dangling_segments(self, backup: bool = True, save: bool = True) -> Dict:
        """
        Remove only the dangling segment portions, preserving connected parts.

        This is more conservative than remove_dangling_tracks() as it tries
        to preserve track segments that have at least one valid connection.

        Args:
            backup: Create backup before modifying
            save: Save changes to file

        Returns:
            Dict with operation results
        """
        if backup:
            self._create_backup()

        dangling = self.find_dangling_tracks()
        trimmed = []
        fully_removed = []

        for d in dangling:
            track = d['track']
            endpoints = d['dangling_endpoints']

            # If both endpoints are dangling, remove entire track
            if len(endpoints) == 2:
                fully_removed.append({
                    'net': d['net'],
                    'layer': pcbnew.LayerName(d['layer']),
                    'reason': 'Both endpoints dangling - removed entire track',
                })
                self.board.Remove(track)
                continue

            # Only one endpoint is dangling - we could potentially trim
            # For now, we just flag it (full implementation would need
            # to trace the track path and find where to cut)
            trimmed.append({
                'net': d['net'],
                'layer': pcbnew.LayerName(d['layer']),
                'dangling_endpoint': endpoints[0]['position'],
                'action': 'flagged_for_review',
                'note': 'Single dangling endpoint - track preserved for manual review',
            })

        if save and fully_removed:
            pcbnew.SaveBoard(str(self.pcb_path), self.board)

        return {
            'success': True,
            'operation': 'trim_dangling_segments',
            'dangling_tracks': len(dangling),
            'fully_removed': len(fully_removed),
            'flagged_for_review': len(trimmed),
            'removed_tracks': fully_removed,
            'flagged_tracks': trimmed,
            'file_saved': save and len(fully_removed) > 0,
            'backup_created': backup,
        }

    def analyze_only(self) -> Dict:
        """
        Analyze dangling tracks without making any changes.

        Returns:
            Dict with analysis results
        """
        dangling = self.find_dangling_tracks()
        connection_points = self._get_connection_points()

        analysis = []
        for d in dangling:
            net = d['net']

            # Find potential extension targets
            potential_targets = []
            if net in connection_points:
                for ep_info in d['dangling_endpoints']:
                    endpoint = ep_info['position']

                    for conn_point in connection_points[net]:
                        dist = self._distance(endpoint, conn_point)
                        if dist > self.CONNECTION_TOLERANCE:
                            potential_targets.append({
                                'position': conn_point,
                                'distance_mm': round(dist / 1_000_000, 3),
                                'within_range': dist <= self.MAX_EXTENSION_DISTANCE,
                            })

                    # Sort by distance
                    potential_targets.sort(key=lambda x: x['distance_mm'])

            analysis.append({
                'net': d['net'],
                'layer': pcbnew.LayerName(d['layer']),
                'dangling_endpoints': [
                    {
                        'position': ep['position'],
                        'position_mm': (
                            round(ep['position'][0] / 1_000_000, 3),
                            round(ep['position'][1] / 1_000_000, 3)
                        ),
                    }
                    for ep in d['dangling_endpoints']
                ],
                'potential_targets': potential_targets[:5],  # Top 5 closest
                'recommendation': 'extend' if potential_targets and potential_targets[0]['within_range'] else 'remove',
            })

        total_tracks = len([t for t in self.board.GetTracks() if t.GetClass() == 'PCB_TRACK'])

        return {
            'success': True,
            'operation': 'analyze',
            'total_tracks': total_tracks,
            'dangling_tracks': len(dangling),
            'dangling_percentage': round(len(dangling) / total_tracks * 100, 1) if total_tracks > 0 else 0,
            'analysis': analysis,
            'summary': {
                'recommend_extend': sum(1 for a in analysis if a['recommendation'] == 'extend'),
                'recommend_remove': sum(1 for a in analysis if a['recommendation'] == 'remove'),
            },
        }

    def _create_backup(self) -> Path:
        """Create a backup of the PCB file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.pcb_path.with_suffix(f'.backup_{timestamp}.kicad_pcb')
        shutil.copy2(self.pcb_path, backup_path)
        return backup_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fix dangling track ends in KiCad PCB files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dangling tracks (no changes)
  python kicad_dangling_track_fixer.py board.kicad_pcb --analyze

  # Remove all dangling tracks
  python kicad_dangling_track_fixer.py board.kicad_pcb --remove

  # Try to extend dangling tracks to nearest connection
  python kicad_dangling_track_fixer.py board.kicad_pcb --extend

  # Conservative trim (remove only fully disconnected)
  python kicad_dangling_track_fixer.py board.kicad_pcb --trim

  # JSON output for scripting
  python kicad_dangling_track_fixer.py board.kicad_pcb --analyze --json
        """,
    )

    parser.add_argument('pcb_path', help='Path to .kicad_pcb file')

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Analyze only, no changes (default)',
    )
    action_group.add_argument(
        '--remove', '-r',
        action='store_true',
        help='Remove tracks with dangling endpoints',
    )
    action_group.add_argument(
        '--extend', '-e',
        action='store_true',
        help='Extend dangling tracks to nearest connection',
    )
    action_group.add_argument(
        '--trim', '-t',
        action='store_true',
        help='Conservative: remove only fully disconnected tracks',
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON',
    )
    parser.add_argument(
        '--max-extension-mm',
        type=float,
        default=5.0,
        help='Maximum distance to extend tracks (mm, default: 5.0)',
    )

    args = parser.parse_args()

    try:
        fixer = DanglingTrackFixer(args.pcb_path)

        # Set max extension distance if specified
        fixer.MAX_EXTENSION_DISTANCE = int(args.max_extension_mm * 1_000_000)

        # Determine operation
        backup = not args.no_backup
        save = not args.dry_run

        if args.remove:
            result = fixer.remove_dangling_tracks(backup=backup, save=save)
        elif args.extend:
            result = fixer.extend_dangling_tracks(backup=backup, save=save)
        elif args.trim:
            result = fixer.trim_dangling_segments(backup=backup, save=save)
        else:
            # Default: analyze only
            result = fixer.analyze_only()

        # Add dry run flag to result
        if args.dry_run:
            result['dry_run'] = True

        # Output
        if args.json:
            # Clean up non-serializable objects
            def clean_for_json(obj):
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()
                            if not k.startswith('track') or k == 'tracks_analyzed'}
                elif isinstance(obj, list):
                    return [clean_for_json(item) for item in obj]
                elif isinstance(obj, tuple):
                    return list(obj)
                elif isinstance(obj, Path):
                    return str(obj)
                else:
                    return obj

            print(json.dumps(clean_for_json(result), indent=2))
        else:
            # Human-readable output
            print(f"\n{'='*60}")
            print(f"Dangling Track Fixer - {result['operation']}")
            print(f"{'='*60}")

            if result['operation'] == 'analyze':
                print(f"Total tracks: {result['total_tracks']}")
                print(f"Dangling tracks: {result['dangling_tracks']} ({result['dangling_percentage']}%)")
                print(f"\nRecommendations:")
                print(f"  - Extend: {result['summary']['recommend_extend']}")
                print(f"  - Remove: {result['summary']['recommend_remove']}")

                if result['analysis']:
                    print(f"\nDetails:")
                    for a in result['analysis'][:10]:  # Show first 10
                        print(f"  Net: {a['net']} ({a['layer']})")
                        print(f"    Recommendation: {a['recommendation']}")
            else:
                print(f"Success: {result['success']}")
                if 'tracks_removed' in result:
                    print(f"Tracks removed: {result.get('tracks_removed', 0)}")
                if 'tracks_extended' in result:
                    print(f"Tracks extended: {result.get('tracks_extended', 0)}")
                if result.get('dry_run'):
                    print("\n[DRY RUN - No changes made]")
                elif result.get('file_saved'):
                    print(f"\nChanges saved to: {args.pcb_path}")

            print(f"{'='*60}\n")

        return 0 if result['success'] else 1

    except FileNotFoundError as e:
        error = {'success': False, 'error': str(e)}
        if args.json:
            print(json.dumps(error, indent=2))
        else:
            print(f"Error: {e}", file=__import__('sys').stderr)
        return 1
    except Exception as e:
        error = {'success': False, 'error': str(e), 'type': type(e).__name__}
        if args.json:
            print(json.dumps(error, indent=2))
        else:
            print(f"Error ({type(e).__name__}): {e}", file=__import__('sys').stderr)
        return 1


if __name__ == '__main__':
    exit(main())
