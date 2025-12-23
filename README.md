# Thesis
All of the code for my Master Thesis.
Model 1 is a basic GMM Model.
Model 2 is a GMM model with RE(Random Effects).
Model 3 is a GMM model with AR(Autoregressive components).
Model 4 is a GMM model wiht AR+RE, which is the most complex model.
There are 8 files in this repository. 4 of them are the very basic code of the 4 models and the other 4 are the ones that I have used to get all of the information that I got from the results.
For example purposes I have made the 4 complete models with different variables that I have used in the thesis. The 1st model showcases the troublemaker variant with tight priors.; The second one the DGP with a different sigma for different groups; The third one showcases the case with tight priors and equal variance with the AR DGP and the 4th model is an example of the troublemaker case with loose priors. 
To achieve any different results from the thesis change the DGP from AR to Trend, change the sigma for any of the groups in the DGP process or relax or tighten the priors in the Stan model.
