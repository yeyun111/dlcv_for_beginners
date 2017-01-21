import numpy.random as random

random.seed(42)

n_tests = 10000

winning_doors = random.randint(0, 3, n_tests)
change_mind_wins = 0
insist_wins = 0

for winning_door in winning_doors:

    first_try = random.randint(0, 3)
    remaining_choices = [i for i in range(3) if i != first_try]
    wrong_choices = [i for i in range(3) if i != winning_door]

    if first_try in wrong_choices:
        wrong_choices.remove(first_try)
    
    screened_out = random.choice(wrong_choices)
    remaining_choices.remove(screened_out)
    
    changed_mind_try = remaining_choices[0]

    change_mind_wins += 1 if changed_mind_try == winning_door else 0
    insist_wins += 1 if first_try == winning_door else 0

print(
    'You win {1} out of {0} tests if you changed your mind\n'
    'You win {2} out of {0} tests if you insist on the initial choice'.format(
        n_tests, change_mind_wins, insist_wins
        )
)
