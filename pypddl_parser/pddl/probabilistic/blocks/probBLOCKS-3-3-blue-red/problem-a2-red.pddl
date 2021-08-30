(define (problem BLOCKS-3-3-blue-red-a2) (:domain blocks-3-3-blue-red-a2)
(:objects
	a - blue
	b - red
	c - green

	(:private
		a2 - redAgent
	)
)
(:init
	(handempty a2)
	(canlift a2 b)
	(clear c)
	(ontable a)
	(on c b)
	(on b a)
)
(:goal
	(and
		(on a b)
		(on b c)
	)
)
)