(define (problem BLOCKS-5-2-blue-red-a1) (:domain blocks-5-2-blue-red-a1)
(:objects
	a - blue
	b - red
	c - blue
	d - red
	e - blue

	(:private
		a1 - blueAgent
	)
)
(:init
	(handempty a1)
	(canlift a1 a)
	(canlift a1 c)
	(canlift a1 e)
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