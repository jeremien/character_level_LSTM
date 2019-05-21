http://karpathy.github.io/2015/05/21/rnn-effectiveness/

# L'efficacité non raisonnable des réseaux de neurones récurrents
Andrej Karpathy

Il y a quelque chose de magique à propos des réseaux de neurones récurrents (RNNs).
Je me rappelle encore quand j'ai entrainé mon premier réseau de neurone récurrent pour légender des images.
Après une douzaine de minutes d'entraînement, mon premier bébé modèle (avec des paramêtres choisis par défaut)
commença à générer de superve description d'images qui commençait à avoir du sens.
Parfois, le rapport entre la simplicité du modèle et la qualité des résultats dépasse toutes tes espérances.
À cette époque, ce qui était choquant était la croyance que les RNNs étaient supposés difficiles à entraîner 
(avec plus d'expérience, il s'avère que ce serait plutôt l'inverse).
Avançons d'une année : je suis en train d'entraîner des RNNs tout le temps et je suis témoin à maintes reprises de leur puissance 
et robustesse, et leurs résultats magiques continuent de m'amuser.
Je partage dans cet article cette magie avec vous 