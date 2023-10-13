(define (domain tiago-domain)
(:predicates
(ROBOT ?x) ; True if x is a robot
(ROOM ?x) (DOOR ?x) (BOOK ?x) (SHELF ?x) (ITEM ?x) (ARM ?x)
(HAS-ARM ?x ?y) ; True if y is an arm of robot x
(HAS-ACCESS ?x ?y) ; True if the room x has access to the door y
(is-at ?x ?y) ; True if x is at room y
(is-open ?x) ; True if the door x is open
(is-free ?x)
(has-item ?x ?y)
)
(:action open-door
:parameters (?robot ?door ?room ?arm)
:precondition (and
    (DOOR ?door)
    (ROOM ?room)
    (ROBOT ?robot)
    (ARM ?arm)
    (HAS-ARM ?robot ?arm)
    (is-at ?robot ?room)
    (HAS-ACCESS ?room ?door)
    (not(is-open ?door)) ;optional
    (is-free ?arm)
)
:effect (is-open ?door)
)
(:action pick-item
:parameters (?robot ?room ?arm ?item)
:precondition (and
    (ROOM ?room)
    (ROBOT ?robot)
    (ARM ?arm)
    (ITEM ?item)
    (HAS-ARM ?robot ?arm)
    (is-at ?robot ?room)
    (is-at ?item ?room)
    (is-free ?arm)
)
:effect (and 
    (has-item ?arm ?item)
    (not(is-free ?arm))
    (not(is-at ?item ?room)) ;the item is at arm, not floor
)
)
(:action put-shelf
:parameters (?robot ?arm ?room ?shelf ?item)
:precondition (and
    (ROOM ?room)
    (ROBOT ?robot)
    (ARM ?arm)
    (ITEM ?item)
    (SHELF ?shelf)
    (HAS-ARM ?robot ?arm)
    (is-at ?robot ?room)
    (is-at ?shelf ?room)    
    (has-item ?arm ?item)
)
:effect (and 
    (not(has-item ?arm ?item))
    (is-free ?arm)
    (has-item ?shelf ?item) ;also works with is-at
    (is-at ?item ?room)
)
)
(:action go-to
:parameters (?robot ?from ?to ?via)
:precondition (and
(ROOM ?from)
(ROOM ?to)
(DOOR ?via)
(is-at ?robot ?from)
(is-open ?via)
(HAS-ACCESS ?from ?via)
(HAS-ACCESS ?to ?via)
)
:effect (and (not (is-at ?robot ?from))
(is-at ?robot ?to))
)
)
