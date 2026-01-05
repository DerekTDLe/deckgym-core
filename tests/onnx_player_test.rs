//! Test ONNX player integration
//! Run with: cargo test --features onnx test_onnx_inference -- --nocapture

#[cfg(feature = "onnx")]
mod tests {
    use deckgym::players::{BatchedOnnxInference, OnnxPlayer};
    use deckgym::rl::{ACTION_SPACE_SIZE, OBSERVATION_SIZE};
    use deckgym::Deck;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_onnx_inference() {
        // Skip if model doesn't exist
        let model_path = "test_model.onnx";
        if !std::path::Path::new(model_path).exists() {
            println!("Skipping test: {} not found", model_path);
            println!("Run Python export first to generate the model");
            return;
        }

        println!("Loading ONNX model from {}...", model_path);
        
        // Test BatchedOnnxInference
        let mut inference = BatchedOnnxInference::new(model_path, false)
            .expect("Failed to load ONNX model");
        
        let mut rng = StdRng::seed_from_u64(42);
        
        // Create dummy observations and masks for 4 environments
        let n_envs = 4;
        let observations: Vec<f32> = (0..n_envs * OBSERVATION_SIZE)
            .map(|i| (i as f32) / (OBSERVATION_SIZE as f32))
            .collect();
        
        // All actions valid
        let action_masks: Vec<bool> = vec![true; n_envs * ACTION_SPACE_SIZE];
        
        println!("Running batch inference on {} environments...", n_envs);
        let actions = inference.predict_batch(&observations, &action_masks, n_envs, &mut rng);
        
        println!("Actions selected: {:?}", actions);
        assert_eq!(actions.len(), n_envs);
        
        // Verify actions are within valid range
        for action in &actions {
            assert!(*action < ACTION_SPACE_SIZE, "Action {} out of range", action);
        }
        
        println!("✓ ONNX inference test passed!");
    }
}
