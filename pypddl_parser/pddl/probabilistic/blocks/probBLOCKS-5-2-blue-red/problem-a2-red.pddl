(define (problem BLOCKS-5-2-blue-red-a2) (:domain blocks-5-2-blue-red-a2)
(:objects
	a - blue
	b - red
	c - blue
	d - red
	e - blue

	(:private
		a2 - redAgent
	)
)
(:init
	(handempty a2)
	(canlift a2 b)
	(canlift a2 d)
	(clear e)
	(ontable a)
	(on e d)
	(on d c)
	(on c b)
	(on b a)
)
(:goal
	(and
		(on a b)
		(on b c)
		(on c d)
		(on d e)
	)
)
)