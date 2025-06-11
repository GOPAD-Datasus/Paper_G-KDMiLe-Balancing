## Base

[(0, 3_157_786), (1, 25170)]
                    pre       rec       spe        f1       geo       iba       sup

0) Não óbito       0.99      1.00      0.24      1.00      0.49      0.26   7368795
    1) Óbito       0.55      0.24      1.00      0.34      0.49      0.22     58105

 avg / total       0.99      0.99      0.25      0.99      0.49      0.26   7426900

Base line: {
    'acc': 0.9925301269708762, 
    'pre': 0.5512865565577291, 
    'rec': 0.24299113673522071, 
    'f1s': 0.337307085861723}

Base line + PCA{
    'acc': 0.9924649584618077, 
    'pre': 0.5400755507349366, 
    'rec': 0.24851561827725668, 
    'f1s': 0.340397444661842}

## Undersampling

#### Random Under Sampler
[(0, 25170), (1, 25170)]
Random Under Sampler: {
    'acc': 0.914261401122945, 
    'pre': 0.06944483713565902, 
    'rec': 0.8031494707856467, 
    'f1s': 0.12783623199918917}
- Best Recall

Random Under Sampler + PCA: {
    'acc': 0.9143808318410104, 
    'pre': 0.06949428356838855, 
    'rec': 0.8025815334308579, 
    'f1s': 0.12791279745128936}

#### Edited Nearest Neighbours
[(0, 3_700_410), (1, 31180)]
Edited Nearest Neighbours: {
    'acc': 0.9923060765595336, 
    'pre': 0.5108423968114572, 
    'rec': 0.3904311160829533, 
    'f1s': 0.44259320678151276}
- Best F1-Score

#### One Sided Selection
[(0, 3769237), (1, 31180)]
One Sided Selection (example data): {
    'acc': 0.9918661024981003, 
    'pre': 0.16666666666666666, 
    'rec': 1.3790060124662143e-05, 
    'f1s': 2.7577838449022366e-05}

## Oversampling

#### Random Oversampler
[(0, 3157786), (1, 3157786)]
Random Oversampler: {
    'acc': 0.9931511128465443, 
    'pre': 0.6425729704179305, 
    'rec': 0.28075036571723605, 
    'f1s': 0.3907679777703253}

Random Oversampler + PCA: {
    'acc': 0.9931610766268564, 
    'pre': 0.6451280015876166, 
    'rec': 0.27973496256776526, 
    'f1s': 0.3902521008403361}
- Best Precision

#### ADASYN
[(0, 3157786), (1, 3160190)]
ADASYN: {
    'acc': 0.9918313158922296, 
    'pre': 0.47252063900503916, 
    'rec': 0.37924447121590227, 
    'f1s': 0.420775253007447}

ADASYN + PCA: {
    'acc': 0.9920493072479769, 
    'pre': 0.489134438305709, 
    'rec': 0.3656828155924619, 
    'f1s': 0.4184942636269634}

## Combining

#### SMOTEENN
[(0, 3789866), (1, 1846094)]
SMOTEENN (example data): {
    'acc': 0.967712958211709, 
    'pre': 0.013632421493874896, 
    'rec': 0.04161840145623035, 
    'f1s': 0.020537597822388568}

---
## Summary
- Random Under Sampler possui o melhor recall
  - É uma técnica que remove registros da classe majoritária
  - Treinamento é feito apenas com dados reais
- Random Over Sampler possui a melhor precisão
  - Gera novos registros para o balanceamento
  - Treinamento misto (Dados reais e gerados)
- Edited Nearest Neighbours possui o melhor f1-score
  - Técnica remove registros de fácil classificação
  - Treinamento é feito apenas com dados reais