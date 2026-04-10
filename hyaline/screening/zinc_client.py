"""
ZINC Database Client
====================

Client for querying the ZINC database (https://zinc.docking.org).

ZINC contains 230M+ purchasable compounds for virtual screening.
"""

import logging
from typing import List, Optional, Dict
from dataclasses import dataclass
import requests
import time

logger = logging.getLogger(__name__)


@dataclass
class ZINCCompound:
    """Compound from ZINC database."""
    zinc_id: str
    smiles: str
    mw: float
    logp: float
    hbd: int  # H-bond donors
    hba: int  # H-bond acceptors
    rotatable_bonds: int
    price_per_mg: float
    supplier: str
    catalog_id: str
    availability: str  # 'in-stock', 'make-on-demand', etc.


class ZINCClient:
    """Client for ZINC database API.
    
    API Documentation: https://zinc.docking.org/api/
    
    Example
    -------
    >>> client = ZINCClient(max_price_per_mg=30.0)
    >>> compounds = client.search_by_substructure("c1ccc2ncccc2c1", max_results=100)
    >>> compounds = client.search_by_similarity("CCO", threshold=0.7, max_results=50)
    """
    
    BASE_URL = "https://zinc.docking.org/api"
    
    def __init__(
        self,
        max_price_per_mg: float = 30.0,
        timeout: int = 30,
    ):
        """
        Parameters
        ----------
        max_price_per_mg : float
            Maximum price per mg in USD
        timeout : int
            Request timeout in seconds
        """
        self.max_price = max_price_per_mg
        self.timeout = timeout
        
    def search_by_substructure(
        self,
        smarts: str,
        max_results: int = 1000,
    ) -> List[ZINCCompound]:
        """Search ZINC by substructure (SMARTS pattern).
        
        Parameters
        ----------
        smarts : str
            SMARTS pattern (e.g., kinase hinge-binding motif)
        max_results : int
            Maximum number of results
            
        Returns
        -------
        List[ZINCCompound]
            Matching compounds
        """
        logger.info(f"Searching ZINC by substructure: {smarts}")
        
        try:
            # ZINC API endpoint for substructure search
            url = f"{self.BASE_URL}/search"
            params = {
                'structure': smarts,
                'structure.search_type': 'substructure',
                'count': max_results,
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            results = []
            data = response.json()
            
            for item in data.get('results', [])[:max_results]:
                try:
                    compound = ZINCCompound(
                        zinc_id=item.get('zinc_id', ''),
                        smiles=item.get('smiles', ''),
                        mw=float(item.get('mw', 0)),
                        logp=float(item.get('logp', 0)),
                        hbd=int(item.get('hbd', 0)),
                        hba=int(item.get('hba', 0)),
                        rotatable_bonds=int(item.get('rb', 0)),
                        price_per_mg=float(item.get('price', 999)),
                        supplier=item.get('supplier', 'unknown'),
                        catalog_id=item.get('catalog_id', ''),
                        availability=item.get('availability', 'unknown'),
                    )
                    results.append(compound)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse compound: {e}")
                    continue
            
            logger.info(f"Found {len(results)} compounds")
            return self.filter_by_price(results)
            
        except requests.RequestException as e:
            logger.error(f"ZINC API request failed: {e}")
            return []
        
    def search_by_similarity(
        self,
        smiles: str,
        threshold: float = 0.7,
        max_results: int = 1000,
    ) -> List[ZINCCompound]:
        """Search ZINC by molecular similarity (Tanimoto).
        
        Parameters
        ----------
        smiles : str
            Query molecule SMILES
        threshold : float
            Tanimoto similarity threshold (0-1)
        max_results : int
            Maximum number of results
            
        Returns
        -------
        List[ZINCCompound]
            Similar compounds
        """
        logger.info(f"Searching ZINC by similarity: {smiles} (threshold={threshold})")
        
        try:
            url = f"{self.BASE_URL}/search"
            params = {
                'structure': smiles,
                'structure.search_type': 'similarity',
                'structure.similarity': threshold,
                'count': max_results,
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            results = []
            data = response.json()
            
            for item in data.get('results', [])[:max_results]:
                try:
                    compound = ZINCCompound(
                        zinc_id=item.get('zinc_id', ''),
                        smiles=item.get('smiles', ''),
                        mw=float(item.get('mw', 0)),
                        logp=float(item.get('logp', 0)),
                        hbd=int(item.get('hbd', 0)),
                        hba=int(item.get('hba', 0)),
                        rotatable_bonds=int(item.get('rb', 0)),
                        price_per_mg=float(item.get('price', 999)),
                        supplier=item.get('supplier', 'unknown'),
                        catalog_id=item.get('catalog_id', ''),
                        availability=item.get('availability', 'unknown'),
                    )
                    results.append(compound)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse compound: {e}")
                    continue
            
            logger.info(f"Found {len(results)} similar compounds")
            return self.filter_by_price(results)
            
        except requests.RequestException as e:
            logger.error(f"ZINC API request failed: {e}")
            return []
        
    def get_compound(self, zinc_id: str) -> Optional[ZINCCompound]:
        """Get compound details by ZINC ID.
        
        Parameters
        ----------
        zinc_id : str
            ZINC ID (e.g., 'ZINC000001234567')
            
        Returns
        -------
        ZINCCompound or None
        """
        logger.info(f"Fetching ZINC compound: {zinc_id}")
        
        try:
            url = f"{self.BASE_URL}/substances/{zinc_id}"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            compound = ZINCCompound(
                zinc_id=data.get('zinc_id', zinc_id),
                smiles=data.get('smiles', ''),
                mw=float(data.get('mw', 0)),
                logp=float(data.get('logp', 0)),
                hbd=int(data.get('hbd', 0)),
                hba=int(data.get('hba', 0)),
                rotatable_bonds=int(data.get('rb', 0)),
                price_per_mg=float(data.get('price', 999)),
                supplier=data.get('supplier', 'unknown'),
                catalog_id=data.get('catalog_id', ''),
                availability=data.get('availability', 'unknown'),
            )
            
            return compound
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch ZINC compound {zinc_id}: {e}")
            return None
        

    def ping(self) -> bool:
        """Check if ZINC API is reachable."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/substances",
                timeout=self.timeout,
                params={"count": 1},
            )
            return response.status_code == 200
        except Exception:
            return False

    def filter_by_price(
        self,
        compounds: List[ZINCCompound],
    ) -> List[ZINCCompound]:
        """Filter compounds by maximum price."""
        return [c for c in compounds if c.price_per_mg <= self.max_price]
