(define (problem BLOCKS-4-3-blue-red-a1) (:domain blocks-4-3-blue-red-a1)
(:objects
	a - blue
	b - red
	c - green
	d - blue

	(:private
		a1 - blueAgent
	)
)
(:init
	(handempty a1)
	(canlift a1 a)
	(canlift a1 d)
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