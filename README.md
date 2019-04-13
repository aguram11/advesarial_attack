# adveserial_attack
Coded from scratch Carlini-Wagner attack on a Gaussian classifier to understand the popular adversarial attacks operation on a Gaussian classifier.

Adverserial attack is a hot topic right now in Machine learning. Companies like Facebook that are involved in lot of image processing to filter out unwanted images, are conducting research in this topic. These kinds of attacks can fool the neural networks or AI employed by these companies to make them completely void and showing the flaws in the neural nets. Thus knowing the flaws can help the companies to improve their neural nets and algorithms.  
Carlini-Wagner attack on a Gaussian classifier to understand the popular adversarial attacks operation on a Gaussian classifier.
The goal is to carry out the CW attack on the two versions of the Gaussian classifier developed in the last project. I am using gradient descent to find the minimum perturbation.
To put in simple words the aim of the attack is perturb the image very slightly so that it essentially looks the same to the eye but the classifier is completely fooled. The target class for the attack is chosen to be grass, so at end the I want rhe classifier to output all black image i.e. the cat is not identified and is misclassified as grass. The classifier will work poorly on the perturbed image produced by the attack. 
