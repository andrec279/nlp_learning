# README
## Part 1
Unlabeled Attachment Score and Labeled Attachment Score:

Validation set: {'uas': 0.5742994241324775, 'las': 0.3490766501447936}

Test set: {'uas': 0.5748405017131534, 'las': 0.3480172399118694}


## Part 2
#### Validation Accuracies

lambda = 0.25: {Rel Pos Acc: 0.6578407488139505 Dep Rel Acc: 0.8404282600333376}

lambda = 0.5: {Rel Pos Acc: 0.6981664315937941 Dep Rel Acc: 0.8288242082318246}

lambda = 0.75: {Rel Pos Acc: 0.7147711245031414 Dep Rel Acc: 0.7960635978971663}


## Part 3
#### Validation UAS / LAS

**Argmax decoding:**

lambda = 0.25: {'uas': 0.7082867045855111, 'las': 0.6717764979708087}

lambda = 0.5: {'uas': 0.7583384351534526, 'las': 0.713303513746909}

lambda = 0.75: {'uas': 0.7717274969384933, 'las': 0.7063934612931998}

**MST Decoding:**

lambda = 0.25: {'uas': 0.5352468026560252, 'las': 0.5089603000367958}

lambda = 0.5: {'uas': 0.5655201755446546, 'las': 0.5330523743437628}

lambda = 0.75: {'uas': 0.5683420736297764, 'las': 0.5202558438733217}


#### Test UAS / LAS

Best lambda for each decoding method was chosen based on LAS performance.

Best Argmax decoding lambda: 0.5 -> {'uas': 0.7746703788440762, 'las': 0.7271580696010198}

Best MST decoding lambda: 0.5 -> {'uas': 0.5833733142783853, 'las': 0.5469186524364584}

