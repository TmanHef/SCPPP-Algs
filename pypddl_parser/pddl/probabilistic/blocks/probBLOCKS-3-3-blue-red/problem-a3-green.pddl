(define (problem BLOCKS-3-3-blue-red-a3) (:domain blocks-3-3-blue-red-a3)
(:objects
	a - blue
	b - red
	c - green

	(:private
		a3 - greenAgent
	)
)
(:init
	(handempty a3)
	(canlift a3 c)
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