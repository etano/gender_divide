# Random

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 370          | 0          |
| male   | 97           | 0          |

- accuracy: 0.50
- female precision: 0.50
- male precision: 0.50
- female recall: 0.50
- male recall: 0.50

# Always female

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 370          | 0          |
| male   | 97           | 0          |

- accuracy: 0.79
- female precision: 1.0
- male precision: 0.0
- female recall: 1.0
- male recall: 0.0

# Naive CNN

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 260          | 110        |
| male   | 55           | 39         |

- accuracy: 0.64
- female precision: 0.83
- male precision: 0.26
- female recall: 0.70
- male recall: 0.41

# VGG16

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 314          | 56         |
| male   | 32           | 65         |

- accuracy: 0.81
- female precision: 0.90
- male precision: 0.54
- female recall: 0.85
- male recall: 0.67

# Face detector + gender CNN

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 88           | 24         |
| male   | 6            | 15         |

- accuracy: 0.77
- female precision: 0.94
- male precision: 0.38
- female recall: 0.24
- male recall: 0.15

# Ensemble

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 324          | 46         |
| male   | 39           | 58         |

- accuracy: 0.82
- female precision: 0.88
- male precision: 0.56
- female recall: 0.88
- male recall: 0.60

# Naive CNN (amazon)

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 297          | 73         |
| male   | 70           | 24         |

- accuracy: 0.69
- female precision: 0.81
- male precision: 0.25
- female recall: 0.80
- male recall: 0.26

# VGG (amazon)

|        | pred. female | pred. male |
| ------ | ------------ | ---------- |
| female | 269          | 101        |
| male   | 45           | 52         |

- accuracy: 0.69
- female precision: 0.86
- male precision: 0.34
- female recall: 0.73
- male recall: 0.54
