# Variational Autoencoder (VAE) vs Autoencoder (AE) Report

## Summary
This experiment implemented and compared a Variational Autoencoder (VAE) and a standard Autoencoder (AE) on the Fashion MNIST dataset. The models were tested with latent dimensions of 2 and 10 to evaluate their impact on reconstruction quality and sample generation.

### Latent Dimension: 2
- VAE Validation Loss: 261.9250
- AE Validation Loss: 253.5848
- VAE Training Losses: ['296.2102', '268.8747', '265.6082', '263.7926', '262.6630', '261.8409', '261.1562', '260.5817', '260.0509', '259.6240']
- AE Training Losses: ['285.7456', '261.6696', '258.0063', '256.1554', '254.9149', '253.8618', '253.1024', '252.5988', '252.1040', '251.5515']

### Latent Dimension: 10
- VAE Validation Loss: 240.0494
- AE Validation Loss: 218.3014
- VAE Training Losses: ['277.3955', '248.1355', '244.4328', '242.4381', '241.2067', '240.2980', '239.6998', '239.1418', '238.7248', '238.4272']
- AE Training Losses: ['262.0768', '225.9598', '222.1071', '220.2454', '219.0875', '218.2032', '217.5248', '216.9758', '216.4791', '216.0655']

## Challenges and Insights
- **Challenges**: Balancing the reconstruction loss and KL-divergence in the VAE was critical. The KL-divergence term sometimes dominated, leading to poorer reconstructions compared to the AE, especially with smaller latent dimensions.
- **Insights**: The VAE with a 2D latent space provided better visualization but poorer reconstruction quality compared to the 10D latent space. The AE consistently achieved lower reconstruction loss due to the absence of the KL-divergence term, but it lacked the generative capabilities of the VAE. Higher latent dimensions improved reconstruction quality for both models but made latent space visualization more challenging, requiring t-SNE.
- **Visualization**: The 2D latent space showed clear class separation for the VAE, while the 10D space required t-SNE for visualization, which still revealed some class clustering. Generated samples from the VAE were more diverse with higher latent dimensions.

Visualizations are saved as 'reconstructions.png', 'generated_samples.png', and 'latent_space.png' (or 'latent_space_tsne.png' for higher dimensions).



