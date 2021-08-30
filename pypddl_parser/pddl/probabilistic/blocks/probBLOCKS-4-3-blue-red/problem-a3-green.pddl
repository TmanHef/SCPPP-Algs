(define (problem BLOCKS-4-3-blue-red-a3) (:domain blocks-4-3-blue-red-a3)
(:objects
	a - blue
	b - red
	c - green
	d - blue

	(:private
		a3 - greenAgent
	)
)
(:init
	(handempty a3)
	(canlift a3 c)
	(clear d)
	(ontable a)
	(on d c)
	(on c b)
	(on b a)
)
(:goal
	(and
		(on a b)
		(on b c)
		(on c d)
	)
)
)