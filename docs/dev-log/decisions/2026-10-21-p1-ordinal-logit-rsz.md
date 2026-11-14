# Polished ordinal logit receipt

Date: 2026-10-21

Status: SIGNED

Shinichi Nakagawa signed this sentence on 2026-10-21: "Record the polished ordinal logit comparison as the receipt for family-ORDINAL-LOGIT-RSZ only."

The receipt is `docs/dev-log/core070/true-parity-latest/receipts/family/campaign/ORDINAL-LOGIT-RSZ.json`.

The numeric bars were not changed. They stay at log-likelihood 1e-6, Lambda Lambda' 0.0001, and thresholds 0.0001. The receipt judges those three quantities. It does not judge a linear predictor or predict.

Both engines meet the stopping rule: a positive-definite Hessian and a max absolute gradient below 1e-5. The polished R gradient is 2.797695408673917e-11 and the minimum eigenvalue is 0.7696554085897079. Julia, measured at the stored parameters, has max absolute gradient 7.0940586738288394e-06 and minimum eigenvalue 0.769168573621117. An earlier polish file reported 2.8194335754960771e-6, also under 1e-5.

The gaps are log-likelihood 1.564330887049437e-10, Lambda Lambda' 1.532503418388842e-6, and thresholds 1.101821489601917e-7. The data sha256 is df6a45d4f51cabe50a394aad203fae7368fd2abb88729d067e7683d2d193c0c1.

Spider, beetle, urbanisation, identity, and the six left rows stay unrecorded. C3, C4, and X2 stay open.
