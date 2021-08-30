(define (problem BLOCKS-6-2-blue-red-a1) (:domain blocks-6-2-blue-red-a1)
(:objects
	a - blue
	b - red
	c - blue
	d - red
	e - blue
	f - red

	(:private
		a1 - blueAgent
	)
)
(:init
	(handempty a1)
	(canlift a1 a)
	(canlift a1 c)
	(canlift a1 e)
	(ontable a)
	(ontable b)
	(clear e)
	(clear f)
	(on e c)
	(on c a)
	(on f d)
	(on d b)

)
(:goal
	(and
		(on a b)
		(on b c)
		(on c d)
		(on d e)
		(on e f)
	)
)
)