#!/usr/bin/env python3
"""
Flask web interface for Paper2Codabench POC.

Shows:
- TaskSpecs extracted from papers
- Generated bundles
- Verification seals

Usage:
    python web/app.py
    Then visit: http://localhost:5000
"""
import json
import sys
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify, send_file

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from seal import VerificationSeal

app = Flask(__name__)

# Initialize seal verifier
seal_verifier = VerificationSeal()


def load_papers_metadata():
    """Load papers metadata from papers/metadata.json"""
    metadata_path = Config.PAPERS_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f).get('competitions', [])
    return []


def load_taskspec(paper_id: str):
    """Load TaskSpec for a paper"""
    taskspec_path = Config.TASKSPEC_DIR / f"{paper_id}.taskspec.json"
    if taskspec_path.exists():
        with open(taskspec_path, 'r') as f:
            return json.load(f)
    return None


def get_bundle_info(paper_id: str):
    """Get information about a bundle"""
    bundle_path = Config.BUNDLES_DIR / paper_id

    if not bundle_path.exists():
        return None

    # Count files
    file_count = len(list(bundle_path.rglob("*")))

    # Get size
    total_size = sum(f.stat().st_size for f in bundle_path.rglob("*") if f.is_file())

    # Check for README
    readme_exists = (bundle_path / "README.md").exists()

    # Get creation time from seal
    seals = load_seals(paper_id)
    creation_time = None
    if seals:
        # Find bundle_creation seal
        for seal in seals:
            if seal.get('seal_type') == 'bundle_creation':
                creation_time = seal.get('timestamp')
                break

    return {
        'exists': True,
        'path': bundle_path,
        'file_count': file_count,
        'size_bytes': total_size,
        'size_mb': total_size / (1024 * 1024),
        'readme_exists': readme_exists,
        'creation_time': creation_time,
    }


def load_seals(paper_id: str):
    """Load all seals for a bundle"""
    bundle_path = Config.BUNDLES_DIR / paper_id
    seals_dir = bundle_path / "seals"

    if not seals_dir.exists():
        return []

    seals = []
    for seal_file in sorted(seals_dir.glob("*.json")):
        try:
            with open(seal_file, 'r') as f:
                seal_data = json.load(f)

            # Verify seal
            is_valid = seal_verifier.verify_seal(seal_data.copy())
            seal_data['verified'] = is_valid
            seal_data['seal_file'] = seal_file.name

            seals.append(seal_data)
        except Exception as e:
            print(f"Error loading seal {seal_file}: {e}")

    return seals


@app.route('/')
def index():
    """Dashboard showing all papers"""
    papers = load_papers_metadata()

    # Enrich with TaskSpec and bundle status
    for paper in papers:
        paper_id = paper['paper_id']

        # Check TaskSpec status
        taskspec = load_taskspec(paper_id)
        paper['taskspec_exists'] = taskspec is not None

        if taskspec:
            paper['task_name'] = taskspec.get('task_name', 'Unknown')
            paper['primary_metric'] = taskspec.get('evaluation', {}).get('primary_metric', 'N/A')

        # Check bundle status
        bundle_info = get_bundle_info(paper_id)
        paper['bundle_exists'] = bundle_info is not None

        if bundle_info:
            paper['bundle_info'] = bundle_info

    return render_template('index.html', papers=papers)


@app.route('/taskspec/<paper_id>')
def view_taskspec(paper_id):
    """View TaskSpec for a paper"""
    taskspec = load_taskspec(paper_id)

    if taskspec is None:
        return f"TaskSpec not found for {paper_id}", 404

    # Get paper metadata
    papers = load_papers_metadata()
    paper_info = next((p for p in papers if p['paper_id'] == paper_id), None)

    return render_template('taskspec.html', taskspec=taskspec, paper_info=paper_info)


@app.route('/bundle/<paper_id>')
def view_bundle(paper_id):
    """View bundle information and seals"""
    bundle_info = get_bundle_info(paper_id)

    if bundle_info is None:
        return f"Bundle not found for {paper_id}", 404

    # Load TaskSpec
    taskspec = load_taskspec(paper_id)

    # Load seals
    seals = load_seals(paper_id)

    # Get paper metadata
    papers = load_papers_metadata()
    paper_info = next((p for p in papers if p['paper_id'] == paper_id), None)

    # Get bundle file tree
    bundle_path = bundle_info['path']
    file_tree = []
    for item in sorted(bundle_path.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(bundle_path)
            file_tree.append({
                'path': str(rel_path),
                'size': item.stat().st_size,
            })

    return render_template(
        'bundle.html',
        paper_id=paper_id,
        paper_info=paper_info,
        taskspec=taskspec,
        bundle_info=bundle_info,
        seals=seals,
        file_tree=file_tree
    )


@app.route('/api/verify_seal/<paper_id>/<seal_file>')
def api_verify_seal(paper_id, seal_file):
    """API endpoint to verify a seal"""
    seal_path = Config.BUNDLES_DIR / paper_id / "seals" / seal_file

    if not seal_path.exists():
        return jsonify({'error': 'Seal not found'}), 404

    try:
        seal_data = seal_verifier.load_seal(seal_path)
        is_valid = seal_verifier.verify_seal(seal_data.copy())

        return jsonify({
            'valid': is_valid,
            'seal_id': seal_data.get('seal_id'),
            'timestamp': seal_data.get('timestamp'),
            'seal_type': seal_data.get('seal_type'),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """API endpoint for overall statistics"""
    papers = load_papers_metadata()

    stats = {
        'total_papers': len(papers),
        'taskspecs_extracted': 0,
        'bundles_generated': 0,
        'total_seals': 0,
    }

    for paper in papers:
        paper_id = paper['paper_id']

        if load_taskspec(paper_id):
            stats['taskspecs_extracted'] += 1

        if get_bundle_info(paper_id):
            stats['bundles_generated'] += 1

        seals = load_seals(paper_id)
        stats['total_seals'] += len(seals)

    return jsonify(stats)


@app.template_filter('format_timestamp')
def format_timestamp(timestamp_str):
    """Format ISO timestamp for display"""
    if not timestamp_str:
        return 'N/A'

    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except:
        return timestamp_str


@app.template_filter('format_size')
def format_size(size_bytes):
    """Format file size for display"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def main():
    print("=" * 60)
    print("Paper2Codabench POC - Web Interface")
    print("=" * 60)
    print(f"\nProject root: {Config.PROJECT_ROOT}")
    print(f"TaskSpecs: {Config.TASKSPEC_DIR}")
    print(f"Bundles: {Config.BUNDLES_DIR}")
    print("\nStarting Flask development server...")
    print("Visit: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
