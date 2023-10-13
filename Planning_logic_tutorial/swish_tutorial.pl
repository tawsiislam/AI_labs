detectives(De) :-
length(De, 3),
member(e(peralta,_,blue), De),
member(e(_,coffee,green),De),
member(e(diaz,tea,_),De),
member(e(holt,_,_),De), % Added one of the missing item used to solve
member(e(_,milk,_),De),
member(e(_,_,white),De).
