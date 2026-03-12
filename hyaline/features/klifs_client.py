"""
KLIFS API Client
================

Reusable client for the KLIFS (Kinase-Ligand Interaction Fingerprints and
Structures) REST API.  Wraps key endpoints for fetching kinase metadata,
structure lists, pocket sequences, and 3-D coordinate files.

Features:
- Automatic rate-limiting (0.2 s between requests)
- Exponential-backoff retry on transient failures
- Local file cache in ``data/klifs_pockets/``
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://klifs.net/api"
CACHE_DIR = Path("data/klifs_pockets")
REQUEST_DELAY = 0.2        # seconds between API calls
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0
TIMEOUT = 30               # request timeout in seconds


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StructureInfo:
    """Metadata for a single KLIFS kinase structure."""
    structure_id: int
    kinase: str
    kinase_id: int
    pdb: str
    chain: str
    pocket: str                # 85-char aligned pocket sequence
    dfg: str = ""
    ac_helix: str = ""
    resolution: float = 0.0
    quality_score: float = 0.0
    missing_residues: int = 0
    missing_atoms: int = 0
    ligand: str = ""
    allosteric_ligand: str = ""
    grich_distance: float = 0.0
    grich_angle: float = 0.0
    grich_rotation: float = 0.0
    # sub-pocket binding booleans
    front: bool = False
    gate: bool = False
    back: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class KLIFSClient:
    """Thin wrapper around the KLIFS REST API with caching & rate limiting."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        cache_dir: str | Path = CACHE_DIR,
        delay: float = REQUEST_DELAY,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._delay = delay
        self._last_request_time: float = 0.0
        self._session = requests.Session()

    # -- internal helpers ---------------------------------------------------

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        raw: bool = False,
    ) -> Any:
        """Issue a GET request with retry + rate limiting.

        Parameters
        ----------
        endpoint : str
            Path relative to ``self.base_url`` (e.g. ``"/kinase_ID"``).
        params : dict, optional
            Query parameters.
        raw : bool
            If *True*, return the raw ``Response`` object (useful for
            binary / MOL2 / PDB downloads).
        """
        url = f"{self.base_url}{endpoint}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()
            try:
                resp = self._session.get(url, params=params, timeout=TIMEOUT)
                self._last_request_time = time.monotonic()
                resp.raise_for_status()
                if raw:
                    return resp
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                wait = BACKOFF_FACTOR ** attempt
                logger.warning(
                    "KLIFS request %s failed (attempt %d/%d): %s – retrying in %.1fs",
                    url, attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"KLIFS request to {url} failed after {MAX_RETRIES} attempts: {last_exc}"
        )

    # -- public API ---------------------------------------------------------

    def get_kinase_id(self, name: str, species: str = "Human") -> Optional[int]:
        """Resolve an HGNC kinase name to a KLIFS kinase ID.

        Returns
        -------
        int or None
            The KLIFS kinase ID, or *None* if not found.
        """
        data = self._get("/kinase_ID", params={"kinase_name": name, "species": species})
        if data and isinstance(data, list) and len(data) > 0:
            return data[0].get("kinase_ID")
        return None

    def get_kinase_info(self, kinase_id: int) -> List[Dict[str, Any]]:
        """Return detailed kinase metadata (Uniprot, family, group, …)."""
        return self._get("/kinase_information", params={"kinase_ID": kinase_id})

    def get_structures(
        self,
        kinase_id: int,
        *,
        max_structures: Optional[int] = None,
    ) -> List[StructureInfo]:
        """Fetch all structures for a kinase, parsed into ``StructureInfo``.

        Parameters
        ----------
        kinase_id : int
            KLIFS kinase ID.
        max_structures : int, optional
            Limit the number of returned structures (sorted by quality_score
            descending).
        """
        raw = self._get("/structures_list", params={"kinase_ID": kinase_id})
        if not raw or not isinstance(raw, list):
            return []

        structures: List[StructureInfo] = []
        for s in raw:
            try:
                info = StructureInfo(
                    structure_id=s["structure_ID"],
                    kinase=s.get("kinase", ""),
                    kinase_id=s.get("kinase_ID", kinase_id),
                    pdb=s.get("pdb", ""),
                    chain=s.get("chain", ""),
                    pocket=s.get("pocket", ""),
                    dfg=s.get("DFG", ""),
                    ac_helix=s.get("aC_helix", ""),
                    resolution=float(s.get("resolution", 0) or 0),
                    quality_score=float(s.get("quality_score", 0) or 0),
                    missing_residues=int(s.get("missing_residues", 0) or 0),
                    missing_atoms=int(s.get("missing_atoms", 0) or 0),
                    ligand=s.get("ligand", ""),
                    allosteric_ligand=s.get("allosteric_ligand", ""),
                    grich_distance=float(s.get("Grich_distance", 0) or 0),
                    grich_angle=float(s.get("Grich_angle", 0) or 0),
                    grich_rotation=float(s.get("Grich_rotation", 0) or 0),
                    front=bool(s.get("front", False)),
                    gate=bool(s.get("gate", False)),
                    back=bool(s.get("back", False)),
                )
                structures.append(info)
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skipping malformed structure entry: %s", exc)

        # Sort best-quality first
        structures.sort(key=lambda si: si.quality_score, reverse=True)

        if max_structures is not None:
            structures = structures[:max_structures]

        return structures

    def get_pocket_mol2(self, structure_id: int) -> str:
        """Download the KLIFS 85-residue pocket in MOL2 format.

        Results are cached to disk at
        ``<cache_dir>/<structure_id>/pocket.mol2``.
        """
        cache_path = self.cache_dir / str(structure_id) / "pocket.mol2"
        if cache_path.exists():
            return cache_path.read_text()

        resp = self._get(
            "/structure_get_pocket",
            params={"structure_ID": structure_id},
            raw=True,
        )
        text = resp.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
        return text

    def get_complex_pdb(self, structure_id: int) -> str:
        """Download the full complex in PDB format.

        Results are cached to disk at
        ``<cache_dir>/<structure_id>/complex.pdb``.
        """
        cache_path = self.cache_dir / str(structure_id) / "complex.pdb"
        if cache_path.exists():
            return cache_path.read_text()

        resp = self._get(
            "/structure_get_pdb_complex",
            params={"structure_ID": structure_id},
            raw=True,
        )
        text = resp.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
        return text

    def get_interaction_fingerprint(self, structure_id: int) -> Optional[str]:
        """Return the 595-bit IFP string for a structure."""
        data = self._get(
            "/interactions_get_IFP",
            params={"structure_ID": structure_id},
        )
        if data and isinstance(data, list) and len(data) > 0:
            return data[0].get("IFP")
        return None

    # -- batch helpers ------------------------------------------------------

    def get_structures_by_pdb(
        self,
        pdb_code: str,
        max_structures: Optional[int] = None,
    ) -> List[StructureInfo]:
        """Get all structures for a given PDB code.
        
        Parameters
        ----------
        pdb_code : str
            PDB code (e.g., '3qri')
        max_structures : int, optional
            Maximum number of structures to return
            
        Returns
        -------
        List[StructureInfo]
            List of structures for this PDB
        """
        endpoint = f"structures_pdb_list"
        params = {"pdb-codes": pdb_code.upper()}
        
        data = self._get(endpoint, params=params)
        if not data:
            return []
        
        structures = []
        for item in data[:max_structures] if max_structures else data:
            try:
                structures.append(StructureInfo(
                    structure_id=item["structure_ID"],
                    kinase=item["kinase"],
                    kinase_id=item["kinase_ID"],
                    pdb=item["pdb"],
                    chain=item["chain"],
                    alternate_model=item.get("alt", ""),
                    pocket=item.get("pocket", ""),
                    dfg=item.get("DFG", ""),
                    ac_helix=item.get("aC_helix", ""),
                    resolution=float(item.get("resolution", 0) or 0),
                    quality_score=float(item.get("qualityscore", 0) or 0),
                    ligand=item.get("ligand", ""),
                    grich_distance=float(item.get("Grich_distance", 0) or 0),
                    grich_angle=float(item.get("Grich_angle", 0) or 0),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed structure: %s", exc)
                continue
        
        return structures

    def fetch_kinase_structures(
        self,
        kinase_name: str,
        *,
        species: str = "Human",
        max_structures: Optional[int] = None,
        download_pdb: bool = False,
    ) -> List[StructureInfo]:
        """Convenience: resolve kinase name → structures, optionally download PDBs."""
        kid = self.get_kinase_id(kinase_name, species=species)
        if kid is None:
            logger.warning("Kinase '%s' not found in KLIFS", kinase_name)
            return []
        logger.info("Kinase %s → KLIFS ID %d", kinase_name, kid)

        structures = self.get_structures(kid, max_structures=max_structures)
        logger.info("  Found %d structures", len(structures))

        if download_pdb:
            for si in structures:
                try:
                    self.get_complex_pdb(si.structure_id)
                except Exception as exc:
                    logger.warning("  Failed to download PDB for %d: %s", si.structure_id, exc)

        return structures


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    client = KLIFSClient()

    kid = client.get_kinase_id("ABL1")
    print(f"ABL1 KLIFS ID: {kid}")

    if kid:
        structs = client.get_structures(kid, max_structures=3)
        for s in structs:
            print(f"  {s.pdb} chain {s.chain} | DFG={s.dfg} αC={s.ac_helix} "
                  f"| quality={s.quality_score:.1f} | pocket[0:20]={s.pocket[:20]}…")
