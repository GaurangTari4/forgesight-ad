"""Create a read-only inventory for all discoverable MVTec categories."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from forgesight_ad.category_workflows import audit_category, discover_categories

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--dataset-root',type=Path,default=ROOT.parent/'mvtec-ad'/'versions'/'2'); parser.add_argument('--no-verify-images',action='store_true'); args=parser.parse_args()
    categories=discover_categories(args.dataset_root); rows=[audit_category(args.dataset_root,category,not args.no_verify_images) for category in categories]
    report={'dataset_root':str(args.dataset_root),'generated_at':datetime.now(timezone.utc).isoformat(),'categories':rows,'category_count':len(rows),'valid_category_count':sum(row['valid'] for row in rows)}
    output=ROOT/'validation_runs'/f"dataset_inventory_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"; output.mkdir(parents=True,exist_ok=False); (output/'inventory.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
