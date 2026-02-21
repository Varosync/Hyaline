# Briscoe Lab: Positional Information & Morphogen Gradients

**Key Papers:** Briscoe & Ericson (1999), Cohen et al. (2015), Balaskas et al. (2012)

---

## Core Concept: Morphogen → Discrete Fates

Sonic Hedgehog (Shh) gradient creates DISCRETE cell types from CONTINUOUS concentration:

```
High Shh → p3 (V3 interneuron) 
Medium Shh → pMN (motor neuron)
Low Shh → p2 (V2 interneuron)
No Shh → p1 (V1 interneuron)
```

---

## The GRN Logic

1. **Concentration thresholds** activate different TFs
2. **Cross-repression** sharpens boundaries
3. **Bistable switches** create discrete outputs

Key insight:
> "The same morphogen concentration can give different outputs depending on DURATION of exposure"

---

## Relevance to Hyaline

| Briscoe Model | Hyaline Model |
|---------------|---------------|
| Shh concentration | SCENIC+ TF activity score |
| Concentration threshold | Spike threshold |
| Cross-repression | Inhibitory connections |
| Discrete fate | Binary activation |

The math is identical:

```python
# Briscoe: Hill function
activation = C^n / (K^n + C^n)

# Hyaline: LIF with threshold
spike = 1 if membrane_potential > threshold else 0
```

Both use **thresholds** to convert continuous inputs to discrete outputs.

---

## Implementation Insight

Briscoe shows that **threshold + dynamics = pattern formation**

Our model:
1. SCENIC+ context sets the threshold
2. Spiking dynamics simulate binding kinetics
3. Synchronization detects successful complex formation
4. Discrete output: TF activates or not
