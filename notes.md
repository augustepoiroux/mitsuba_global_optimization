# Notes

- calcul analytique pour vérifier le bon comportement du code de calcul de variance

- variance utile pour avoir des informations locales sur létat de la convergence

- mc with gradient descent
- use gradient information to see if a subset of parameters converged
compute moments of the gradient -> variance in the current rendering setting
-> augment the parameters?

observations:

- depending on the spp parameter, there exist sometimes converged parameters that have a better loss than the reference parameters
- global optimizers really slow to converge

Done:

- increasing spp
- scène anisotropique avec 3 lumières (rouge, orange et verte au centre)
- implémenter des méthodes d'optimisation globales -> regarder librairies existantes
- trouver d'autres points de départ pour la scène 1 qui amèneraient à d'autres minimums locaux

  - trouver d'autres procédures de génération de points aléatoires

- roughness 1 (without envmap):

  - interpoler loss entre solutions convergées
    - linear interp 1d

- essayer plusieurs vues pour la scène 1
- upscale textures durant l'optimisation
- texture + envmap sur mesh (ex: vache, pour illustrer meshs homéomorphiques à des sphères)
- rotation/translation de meshs
- scene with non-optimal local minima (trap for gradient descent)

WIP:

- optimiser une sdf (+ robuste que des meshs à optimiser)
- optimisation globale
- littérature sur les optimisations pour inverse rendering
- tester différents algorithmes d'optimisation globale
- différentes stratégies pour le nombre d'itérations pour la descente de gradient
- BO-leap: line 17 of pseudo-code is not clear

TODO:

- parallélisation
-mettre des couleurs aux lapins
- optimisations avec sphères
- double caustiques
- exemple papier wasserstein
- exemples avec différents nombres de dimension (ordres de grandeur)
  - tableaux avec les algorithmes qui marchent bien par ordre de grandeur de dimension
- exemples avec différentes tailles d'images

- montrer des baselines

  - gradient descent + plusieurs points aléatoires
  - random search

- ambiguités de certain problème -> nombreux minimums locaux

- scène avec caustiques
- utiliser la vache la vraie
- multiview avec envmap à optimiser

- gradient index optics

- scène avec objet volumétrique
- optimiser extinction liquide

- roughness 1 (without envmap):
  - interpoler loss entre solutions convergées
    - par texel
    - déterminer les raisons de ces minimums locaux
- upscale la taille du rendu durant l'optimisation

Unrelated:

- quand on cherche thèse dans un labo, regarder liste postdoc
- laurent belcour - unity thèse de master
