(define (problem BLOCKS-4-3-blue-red-a2) (:domain blocks-4-3-blue-red-a2)
(:objects
	a - blue
	b - red
	c - green
	d - blue

	(:private
		a2 - redAgent
	)
)
(:init
	(handempty a2)
	(canlift a2 b)
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