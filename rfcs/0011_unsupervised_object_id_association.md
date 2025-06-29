# RFC 0011: Unsupervised Object ID Association for Cross-Modal Learning

## Summary

This RFC proposes implementing unsupervised object ID association mechanisms that enable learning modules to discover correspondences between their internal object representations without requiring predefined object labels. This addresses the current limitation where voting requires hardcoded object IDs shared across learning modules.

## Motivation

Currently, Monty's voting mechanism relies on all learning modules sharing the same object ID for any given object (e.g., "cup"), which acts as a supervised learning signal. In truly unsupervised scenarios, each learning module creates unique object IDs (e.g., `visual_object_1`, `touch_object_1`) for the same physical object, preventing effective cross-modal voting and consensus building.

This limitation contradicts the goal of unsupervised learning and prevents Monty from working in real-world scenarios where object boundaries and identities are not predefined.

## Detailed Design

### Phase 1: Association Memory and Co-occurrence Learning

1. **Association Memory Structure**
   - Each LM maintains `association_memory`: `{my_object_id: {other_lm_id: {other_object_id: association_data}}}`
   - `association_data` includes: co-occurrence count, spatial consistency scores, temporal context

2. **Co-occurrence Detection**
   - Track when multiple LMs have high evidence for their respective objects simultaneously
   - Record spatial and temporal context of co-occurrences
   - Build confidence scores for object ID associations

### Phase 2: Modified Voting Protocol

1. **Enhanced Vote Messages**
   - Include sender LM ID, confidence scores, spatial context
   - Send association metadata alongside traditional vote data

2. **Probabilistic Vote Mapping**
   - Map incoming votes to local object IDs using learned associations
   - Weight votes by association confidence
   - Handle uncertainty in associations gracefully

### Phase 3: Consensus Building

1. **Multi-Modal Hypothesis Clustering**
   - Group hypotheses from different LMs based on spatial/temporal consistency
   - Use clustering to identify likely same-object hypotheses

2. **Dynamic Association Updates**
   - Continuously refine associations as more evidence is gathered
   - Implement feedback mechanisms for association validation

## Implementation Plan

### Step 1: Create Association Mixin
- `UnsupervisedAssociationMixin` class
- Association memory management
- Co-occurrence tracking algorithms

### Step 2: Extend Learning Module Classes
- Modify `EvidenceGraphLM` to include association capabilities
- Update voting methods (`send_out_vote`, `receive_votes`)
- Add association learning logic

### Step 3: Enhanced Voting Protocol
- Modify vote data structures
- Implement probabilistic vote mapping
- Add association confidence calculations

### Step 4: Experimental Validation
- Create test scenarios for cross-modal learning
- Validate association learning performance
- Compare with current supervised approach

## Benefits

1. **Truly Unsupervised Learning**: No predefined object labels required
2. **Cross-Modal Capability**: Enables association across different sensory modalities
3. **Scalable**: Works with arbitrary numbers of LMs and modalities
4. **Biologically Plausible**: Mimics cortical column association mechanisms
5. **Robust**: Handles uncertainty and incorrect initial associations

## Risks and Mitigation

1. **Computational Overhead**: Mitigated by efficient data structures and optional features
2. **Memory Requirements**: Managed through configurable memory limits and pruning
3. **Convergence Time**: Addressed through smart initialization and confidence thresholds
4. **False Associations**: Prevented through validation mechanisms and confidence scoring

## Testing Strategy

1. **Unit Tests**: Association memory, vote mapping, confidence calculations
2. **Integration Tests**: Multi-LM scenarios with known ground truth
3. **Performance Tests**: Computational overhead and memory usage
4. **Validation Experiments**: Cross-modal object recognition tasks

## Future Extensions

1. **Hierarchical Associations**: Part-whole relationships across modalities
2. **Temporal Sequence Learning**: Dynamic object and scene understanding
3. **Language Grounding**: Associate learned words with grounded objects
4. **Advanced Clustering**: More sophisticated hypothesis grouping algorithms

## Conclusion

This implementation will enable Monty to work in truly unsupervised scenarios, making it more applicable to real-world applications where object identities must be discovered rather than predefined.
