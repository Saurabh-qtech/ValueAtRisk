# check if any VaR, ES or other risk measures satisfies the 4 conditions of coherent risk measure
# 4 conditions are :
# condition 1 Subadditivity : RM(P1) + RM(P2) >= RM(P1 + P2)
# condition 2 Monotonicity : if P1 always performs better than P2 => 
# condition 3 positive Homogeneity : RM( c * P ) = c * RM(P)
# condition 4 Translation Invariance: RM(P + Cash) = RM(P) + Cash

# Sub-additivity and positive Homogeneity are replaced with the convexity condition
# RM(a.P1 + (1 - a).P2) <= a.RM(P1) + (1-a).RM(P2) 