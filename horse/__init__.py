"""Horse platform — Stage 2 classifier.

embed.py    -> frozen CLIP image encoder
model.py    -> small trainable head (embedding -> horseness score)
score.py    -> wires the two together: image -> embedding -> score
anchors.py  -> fixed operator-controlled seed set for the anchor loss
storage.py  -> SQLite storage for community ratings + their embeddings
rate.py     -> record a community rating (embed once, store it)
train.py    -> retrain the head: anchor loss + recency-weighted rating loss
gate.py     -> score >= threshold pass/reject decision (same head/score
               as the rating signal -- see CLAUDE.md "unified drift head")
"""
