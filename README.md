# PGM_project
Study of several papers about "Denoising score matching for diffusion models"

**TODO** : 
1. Authors' repository
   - [x] Run the authors' code
   - [x] Repeat the experiments with the pretrained models

2. Toy experiment (mixture of Gaussians)
   - [x] Repeat experiment with a simple score network
   - [x] Repeat experiment with a similar noise conditional score network
   - [x] Visualize and compare real and estimated scores in both cases
   - [x] Compare Langevin dynamics samples with annealed Langevin dynamics samples

3. MNIST
   - [x] Implement, train and test a noise conditional score network on the MNIST dataset
         
   ***Observation***: Taking too many steps in the annealed Langevin dynamics result in bad samples. Can we explain why?

4. Experiments with parameters (on toy dataset, on MNIST or using the pretrained model)
   - [ ] Compare sliced score matching with denoising score matching (in terms of performance and computational costs)
   - [x] The choice of the coefficients $\lambda(\sigma_i)$
   - [ ] The choice of the coefficients $\alpha(\sigma_i)$
   - [x] The choice of $\epsilon$
   - [ ] The number of steps in the Langevin dynamics (additionally, try taking different number of steps for different $\sigma$)
   - [x] Experiment with different sets $\lbrace \sigma_i \rbrace_{i = 1}^L$

6. Applications
   - [x] Impainting: crop a random part of the image and generate the missing part

7. Theory
   - [x] Discuss the article "Generative modeling by estimating gradients of the data distribution"
   - [x] Discuss the article "A connection between score matching and denoising autoencoders"
   - [ ] Explain connections between "Generative modeling by estimating gradients of the data distribution" and  "A connection between score matching and denoising autoencoders"
   - [ ] Study the article "Denoising Diffusion Probabilistic Models"
   - [ ] Explain connections between "Generative modeling by estimating gradients of the data distribution" and "Denoising Diffusion Probabilistic Models"
   - [ ] Study the article "Score-Based Generative Modeling through Stochastic Differential Equations"
   - [ ] Study the article "Improved Techniques for Training Score-Based Generative Models"

9. Extensions / New experiments
   - [ ] Read about a Metropolis-Hastings update and add it if possible
   - [ ] Experiment with the choice of random projections in the sliced score matching
   - [ ] Try other than Normal noise distributions
   - [ ] In the authors implementation, the estimation of the unified objective is done by sampling random sigmas uniformly and taking the mean over the batch. Does it give an unbiased estimator of the theoretic objective? If not, try to change the procedure and compare the results
   - [ ] Try different architectures
   - [ ] Can we improve the quality of generation by utilizing class labels? Can we address the issue of slow mixing by using class conditional networks?
   
10. Diffusion model (?)
    - [ ] Implement model from the "Denoising Diffusion Probabilistic Models" paper
    - [ ] Compare to the noise conditional score network
