#!/usr/bin/env python3
"""
GPU-Accelerated KLIFS Data Pipeline
Based on my-agents coder guidance.

Downloads kinase structures, extracts features, creates conformational pairs.
"""
import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from collections import defaultdict
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


@dataclass
class PipelineConfig:
    klifs_base_url: str = "https://klifs.net/api_v2"
    max_concurrent: int = 10
    requests_per_second: float = 5.0
    timeout: int = 30
    cache_dir: Path = field(default_factory=lambda: Path("./klifs_cache"))
    species: str = "Human"
    min_resolution: float = 3.5
    
    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class DFGConformation(Enum):
    IN = "in"
    OUT = "out"
    OUT_LIKE = "out-like"
    NA = "na"
    
    @classmethod
    def from_string(cls, s: Optional[str]) -> 'DFGConformation':
        if not s:
            return cls.NA
        s = s.lower().strip()
        if s == 'in':
            return cls.IN
        elif s == 'out':
            return cls.OUT
        elif 'out' in s:
            return cls.OUT_LIKE
        return cls.NA


@dataclass
class KinaseData:
    kinase_id: int
    name: str
    gene_name: str
    family: str
    group: str
    uniprot: str


@dataclass  
class StructureData:
    structure_id: int
    kinase_id: int
    pdb: str
    chain: str
    dfg: DFGConformation
    chelix: str
    pocket_seq: str
    resolution: Optional[float]
    quality: Optional[float]
    ligand: Optional[str]


class RateLimiter:
    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self.last = 0.0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            wait = self.interval - (now - self.last)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last = time.monotonic()


class KLIFSClient:
    """Async KLIFS API client with rate limiting."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.limiter = RateLimiter(config.requests_per_second)
        self.sem = asyncio.Semaphore(config.max_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def _get(self, endpoint: str, params: Dict = None) -> Any:
        url = f"{self.config.klifs_base_url}/{endpoint}"
        async with self.sem:
            await self.limiter.acquire()
            try:
                async with self.session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    logger.warning(f"API {resp.status} for {endpoint}")
                    return None
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return None
    
    async def get_all_kinases(self) -> List[KinaseData]:
        """Fetch all human kinases."""
        logger.info("Fetching all kinases...")
        data = await self._get("kinase_names", {"species": self.config.species})
        if not data:
            return []
        
        kinases = []
        for item in data:
            kinases.append(KinaseData(
                kinase_id=item.get('kinase_ID', 0),
                name=item.get('name', ''),
                gene_name=item.get('gene_name', ''),
                family=item.get('family', ''),
                group=item.get('group', ''),
                uniprot=item.get('accession', '')
            ))
        logger.info(f"Found {len(kinases)} kinases")
        return kinases
    
    async def get_structures(self, kinase_id: int) -> List[StructureData]:
        """Fetch structures for a kinase."""
        data = await self._get("structures_list", {"kinase_ID": [kinase_id]})
        if not data:
            return []
        
        structures = []
        for item in data:
            res = item.get('resolution')
            if res and float(res) > self.config.min_resolution:
                continue
            
            structures.append(StructureData(
                structure_id=item.get('structure_ID', 0),
                kinase_id=kinase_id,
                pdb=item.get('pdb', ''),
                chain=item.get('chain', ''),
                dfg=DFGConformation.from_string(item.get('DFG')),
                chelix=item.get('aC_helix', ''),
                pocket_seq=item.get('pocket', ''),
                resolution=float(res) if res else None,
                quality=item.get('quality_score'),
                ligand=item.get('ligand')
            ))
        return structures
    
    async def get_conformations(self, structure_ids: List[int]) -> Dict[int, Dict]:
        """Get conformational details for structures."""
        if not structure_ids:
            return {}
        
        data = await self._get("structure_conformation", {"structure_ID": structure_ids[:50]})
        if not data:
            return {}
        
        return {item['structure_ID']: item for item in data}


async def download_klifs_dataset(config: PipelineConfig) -> Dict[str, Any]:
    """Download complete KLIFS dataset with conformational data."""
    
    cache_file = config.cache_dir / "klifs_dataset.json"
    
    # Check cache
    if cache_file.exists():
        logger.info(f"Loading from cache: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)
    
    dataset = {
        'kinases': [],
        'structures': [],
        'conformational_pairs': [],
        'stats': {}
    }
    
    async with KLIFSClient(config) as client:
        # Get all kinases
        kinases = await client.get_all_kinases()
        dataset['kinases'] = [{'id': k.kinase_id, 'name': k.name, 'family': k.family} 
                              for k in kinases]
        
        # Get structures for each kinase
        all_structures = []
        kinases_with_both = []
        
        for i, kinase in enumerate(kinases):
            if i % 50 == 0:
                logger.info(f"Processing kinase {i+1}/{len(kinases)}: {kinase.name}")
            
            structures = await client.get_structures(kinase.kinase_id)
            
            # Count DFG conformations
            dfg_in = [s for s in structures if s.dfg == DFGConformation.IN]
            dfg_out = [s for s in structures if s.dfg in (DFGConformation.OUT, DFGConformation.OUT_LIKE)]
            
            if dfg_in and dfg_out:
                kinases_with_both.append({
                    'kinase_id': kinase.kinase_id,
                    'name': kinase.name,
                    'n_dfg_in': len(dfg_in),
                    'n_dfg_out': len(dfg_out),
                    'structures_in': [s.pdb for s in dfg_in[:3]],
                    'structures_out': [s.pdb for s in dfg_out[:3]]
                })
            
            for s in structures:
                all_structures.append({
                    'structure_id': s.structure_id,
                    'kinase_id': kinase.kinase_id,
                    'kinase_name': kinase.name,
                    'pdb': s.pdb,
                    'dfg': s.dfg.value,
                    'chelix': s.chelix,
                    'pocket_seq': s.pocket_seq,
                    'resolution': s.resolution,
                    'ligand': s.ligand
                })
        
        dataset['structures'] = all_structures
        dataset['conformational_pairs'] = kinases_with_both
        dataset['stats'] = {
            'total_kinases': len(kinases),
            'total_structures': len(all_structures),
            'kinases_with_both_conformations': len(kinases_with_both),
            'dfg_in_count': sum(1 for s in all_structures if s['dfg'] == 'in'),
            'dfg_out_count': sum(1 for s in all_structures if s['dfg'] in ['out', 'out-like']),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Saved dataset to {cache_file}")
    
    return dataset


def create_training_features(dataset: Dict, device=DEVICE) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Create training features from KLIFS dataset.
    Uses GPU for feature computation.
    """
    structures = dataset['structures']
    pairs = dataset['conformational_pairs']
    
    # For kinases with both conformations, create feature pairs
    features = []
    labels = []  # Binary: is this DFG-out?
    pdb_ids = []
    
    # Amino acid encoding
    AA_MAP = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY-_')}
    
    for struct in structures:
        if struct['dfg'] not in ['in', 'out', 'out-like']:
            continue
        
        pocket = struct['pocket_seq']
        if not pocket:
            continue
        
        # Encode pocket sequence
        encoded = [AA_MAP.get(aa, 21) for aa in pocket[:85]]
        while len(encoded) < 85:
            encoded.append(21)
        
        features.append(encoded)
        labels.append(1.0 if struct['dfg'] in ['out', 'out-like'] else 0.0)
        pdb_ids.append(struct['pdb'])
    
    if not features:
        return None, None, []
    
    # Convert to GPU tensors
    X = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    
    logger.info(f"Created {len(features)} training samples on {device}")
    logger.info(f"  DFG-in: {int((y == 0).sum())}, DFG-out: {int((y == 1).sum())}")
    
    return X, y, pdb_ids


async def main():
    """Main pipeline execution."""
    print("="*70)
    print("KLIFS DATA PIPELINE")
    print("="*70)
    
    config = PipelineConfig()
    
    # Download dataset
    dataset = await download_klifs_dataset(config)
    
    # Print stats
    stats = dataset['stats']
    print(f"\nDataset Statistics:")
    print(f"  Total kinases: {stats['total_kinases']}")
    print(f"  Total structures: {stats['total_structures']}")
    print(f"  Kinases with both DFG conformations: {stats['kinases_with_both_conformations']}")
    print(f"  DFG-in structures: {stats['dfg_in_count']}")
    print(f"  DFG-out structures: {stats['dfg_out_count']}")
    
    # Show top kinases with both conformations
    pairs = dataset['conformational_pairs']
    print(f"\nTop kinases with both DFG-in and DFG-out:")
    for p in sorted(pairs, key=lambda x: x['n_dfg_in'] + x['n_dfg_out'], reverse=True)[:10]:
        print(f"  {p['name']}: {p['n_dfg_in']} in, {p['n_dfg_out']} out")
    
    # Create training features
    X, y, pdbs = create_training_features(dataset)
    if X is not None:
        print(f"\nTraining data ready: {X.shape}")
    
    return dataset


if __name__ == "__main__":
    asyncio.run(main())
