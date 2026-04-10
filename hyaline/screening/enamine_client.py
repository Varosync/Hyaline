"""
Enamine REAL Database Client
=============================

Client for querying Enamine REAL database (https://enamine.net/compound-collections/real-compounds).

Enamine REAL contains 6.5B+ make-on-demand compounds.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class EnamineCompound:
    """Compound from Enamine REAL database."""
    enamine_id: str
    smiles: str
    mw: float
    logp: float
    hbd: int
    hba: int
    rotatable_bonds: int
    price_per_mg: float
    lead_time_days: int
    availability: str  # 'make-on-demand', 'in-stock'


class EnamineClient:
    """Client for Enamine REAL database.
    
    API Documentation: https://enamine.net/compound-collections/real-compounds/real-database-api
    
    Example
    -------
    >>> client = EnamineClient(max_price_per_mg=30.0)
    >>> compounds = client.search_by_substructure("c1ccc2ncccc2c1", max_results=100)
    """
    
    BASE_URL = "https://enamine.net/api/real"
    
    def __init__(
        self,
        max_price_per_mg: float = 30.0,
        max_lead_time_days: int = 30,
        timeout: int = 30,
    ):
        """
        Parameters
        ----------
        max_price_per_mg : float
            Maximum price per mg in USD
        max_lead_time_days : int
            Maximum synthesis lead time in days
        timeout : int
            Request timeout in seconds
        """
        self.max_price = max_price_per_mg
        self.max_lead_time = max_lead_time_days
        self.timeout = timeout
        
    def search_by_substructure(
        self,
        smarts: str,
        max_results: int = 1000,
    ) -> List[EnamineCompound]:
        """Search Enamine REAL by substructure.
        
        Parameters
        ----------
        smarts : str
            SMARTS pattern
        max_results : int
            Maximum number of results
            
        Returns
        -------
        List[EnamineCompound]
            Matching compounds
        """
        logger.info(f"Searching Enamine REAL by substructure: {smarts}")
        
        try:
            # Enamine REAL API endpoint
            url = f"{self.BASE_URL}/search/substructure"
            payload = {
                'smarts': smarts,
                'limit': max_results,
            }
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            results = []
            data = response.json()
            
            for item in data.get('compounds', [])[:max_results]:
                try:
                    compound = EnamineCompound(
                        enamine_id=item.get('id', ''),
                        smiles=item.get('smiles', ''),
                        mw=float(item.get('mw', 0)),
                        logp=float(item.get('logp', 0)),
                        hbd=int(item.get('hbd', 0)),
                        hba=int(item.get('hba', 0)),
                        rotatable_bonds=int(item.get('rotatable_bonds', 0)),
                        price_per_mg=float(item.get('price_per_mg', 999)),
                        lead_time_days=int(item.get('lead_time_days', 30)),
                        availability=item.get('availability', 'make-on-demand'),
                    )
                    results.append(compound)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse compound: {e}")
                    continue
            
            logger.info(f"Found {len(results)} compounds")
            return self.filter_by_price_and_leadtime(results)
            
        except requests.RequestException as e:
            logger.error(f"Enamine API request failed: {e}")
            # Enamine API may not be publicly accessible - return empty
            logger.warning("Enamine REAL requires license or catalog download")
            return []
        
    def search_by_similarity(
        self,
        smiles: str,
        threshold: float = 0.7,
        max_results: int = 1000,
    ) -> List[EnamineCompound]:
        """Search Enamine REAL by molecular similarity.
        
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
        List[EnamineCompound]
            Similar compounds
        """
        logger.info(f"Searching Enamine REAL by similarity: {smiles} (threshold={threshold})")
        
        try:
            url = f"{self.BASE_URL}/search/similarity"
            payload = {
                'smiles': smiles,
                'threshold': threshold,
                'limit': max_results,
            }
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            results = []
            data = response.json()
            
            for item in data.get('compounds', [])[:max_results]:
                try:
                    compound = EnamineCompound(
                        enamine_id=item.get('id', ''),
                        smiles=item.get('smiles', ''),
                        mw=float(item.get('mw', 0)),
                        logp=float(item.get('logp', 0)),
                        hbd=int(item.get('hbd', 0)),
                        hba=int(item.get('hba', 0)),
                        rotatable_bonds=int(item.get('rotatable_bonds', 0)),
                        price_per_mg=float(item.get('price_per_mg', 999)),
                        lead_time_days=int(item.get('lead_time_days', 30)),
                        availability=item.get('availability', 'make-on-demand'),
                    )
                    results.append(compound)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse compound: {e}")
                    continue
            
            logger.info(f"Found {len(results)} similar compounds")
            return self.filter_by_price_and_leadtime(results)
            
        except requests.RequestException as e:
            logger.error(f"Enamine API request failed: {e}")
            logger.warning("Enamine REAL requires license or catalog download")
            return []
        

    def ping(self) -> bool:
        """Check if Enamine API is reachable."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/status",
                timeout=self.timeout,
            )
            return response.status_code == 200
        except Exception:
            return False

    def filter_by_price_and_leadtime(
        self,
        compounds: List[EnamineCompound],
    ) -> List[EnamineCompound]:
        """Filter compounds by price and lead time."""
        return [
            c for c in compounds
            if c.price_per_mg <= self.max_price
            and c.lead_time_days <= self.max_lead_time
        ]
