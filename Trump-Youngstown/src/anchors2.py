"""Minimal-pair anchors: identical subject matter, opposite holding.

The first attempt (anchors.py) built the axis from thematically opposed
phrases.  That axis turned out to track how *much* a paragraph talks about
executive power rather than whether it endorses it: it put Jackson's "lowest
ebb" paragraph on the autonomy side and filled the constraint pole with
statutory citation boilerplate.

Minimal pairs are the standard remedy.  Each pair holds topic, register, and
vocabulary fixed and flips only the stance, so the difference of the two means
cancels the shared topical direction.
"""

PAIRS = [
    ("The President's claim of exclusive authority over this subject must be sustained.",
     "The President's claim of exclusive authority over this subject must be rejected."),
    ("Congress may not interfere with the President's exercise of this power.",
     "Congress may regulate the President's exercise of this power."),
    ("The President is immune from judicial process for these official acts.",
     "The President is subject to judicial process for these official acts."),
    ("This executive action fell within the President's constitutional authority.",
     "This executive action exceeded the President's constitutional authority."),
    ("The courts must defer to the President's judgment in this area.",
     "The courts must independently review the President's judgment in this area."),
    ("The Constitution commits this decision to the President alone.",
     "The Constitution commits this decision to Congress, not to the President."),
    ("We uphold the President's seizure of property as a valid exercise of executive power.",
     "We hold the President's seizure of property invalid as beyond executive power."),
    ("The President may act in this field without any statutory authorization.",
     "The President may not act in this field without statutory authorization."),
    ("Presidential immunity from prosecution is required by the separation of powers.",
     "Presidential immunity from prosecution finds no support in the separation of powers."),
    ("The President's motives in taking this action are not subject to judicial inquiry.",
     "The President's motives in taking this action are subject to judicial inquiry."),
    ("Statutory limits on the President's power to remove officers are unconstitutional.",
     "Statutory limits on the President's power to remove officers are constitutional.",),
    ("Because the Executive requires dispatch, his discretion here is unreviewable.",
     "Because no officer is above the law, his discretion here is reviewable."),
    ("The statute cannot constrain the President in the exercise of this function.",
     "The statute lawfully constrains the President in the exercise of this function."),
    ("Recognizing such presidential power is faithful to the constitutional design.",
     "Recognizing such presidential power would betray the constitutional design."),
    ("The Executive's determination in this matter binds the courts.",
     "The Executive's determination in this matter does not bind the courts."),
    ("The President need not justify this decision to Congress.",
     "The President must justify this decision to Congress."),
]

AUTONOMY = [a for a, _ in PAIRS]
CONSTRAINT = [b for _, b in PAIRS]
