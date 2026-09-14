"""Anchor phrases defining the constraint <-> autonomy axis.

Fixed BEFORE looking at any results.  Phrases rather than single words, because
single-word embeddings are dominated by topic and these need to sit in the same
register as the corpus.  None of the anchors is copied from Youngstown or from
Trump v. United States, so neither case gets an unearned advantage: the one
canonical phrase each opinion is famous for ("lowest ebb", "conclusive and
preclusive") is deliberately paraphrased rather than quoted.
"""

# Pole A: presidential power as checked, reviewable, and subordinate to law.
CONSTRAINT = [
    "the President's authority is at its weakest when he acts against the will of Congress",
    "Congress has by statute expressly limited the authority of the Executive",
    "the Executive Branch remains accountable to Congress and to the courts",
    "judicial review of executive action is essential to the rule of law",
    "the President is not above the law and must answer to it like any citizen",
    "the separation of powers guards against the concentration of authority in one branch",
    "this constitutional authority belongs to Congress alone and not to the President",
    "the Executive must conform to the limits that Congress has imposed by legislation",
    "checks and balances operate to constrain the discretion of the Executive",
    "an unchecked Executive threatens the liberty the Constitution was designed to secure",
    "the courts have a duty to say what the law is even as against the Executive",
    "emergency does not enlarge constitutional power or suspend its limitations",
]

# Pole B: presidential power as exclusive, independent, and insulated.
AUTONOMY = [
    "the President's authority in this sphere is exclusive and beyond congressional control",
    "this power is committed by the Constitution solely to the Executive",
    "Congress may not regulate the President's exercise of his core constitutional functions",
    "the President must be free to act boldly and without hesitation",
    "as Commander in Chief the President holds exclusive authority over military affairs",
    "courts may not inquire into the motives behind the President's official acts",
    "the Executive requires energy, unity, independence, and dispatch",
    "the President enjoys immunity from suit for acts within his official responsibilities",
    "the President must be able to discharge his duties without fear of later prosecution",
    "the matter is committed to the unreviewable discretion of the Executive",
    "the vesting of executive power confers authority the other branches cannot diminish",
    "subjecting the President to such scrutiny would fatally impair the Presidency",
]
