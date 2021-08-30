(define (problem BLOCKS-3-3-blue-red-a1) (:domain blocks-3-3-blue-red-a1)
(:objects
	a - blue
	b - red
	c - green

	(:private
		a1 - blueAgent
	)
)
(:init
	(handempty a1)
	(canlift a1 a)
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