(define (problem tiago-problem)
(:objects
tiago ; the robot
tiarm1 ; it’s arm
door1
room1
room2
shelf1
book1
)
(:init
(ROBOT tiago)
(HAS-ARM tiago tiarm1)
(DOOR door1)
(ITEM book1)
(BOOK book1)
(ARM tiarm1)
(ROOM room1)
(ROOM room2)
(SHELF shelf1)
(HAS-ACCESS room1 door1)
(HAS-ACCESS room2 door1)
(is-at book1 room1)
(is-at tiago room1)
(is-at shelf1 room2)
(is-free tiarm1)
)
(:goal
(and (is-at tiago room1) (has-item shelf1 book1))
)
)
