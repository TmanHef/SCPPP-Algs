(define (problem BLOCKS-2-2-blue-red-a1) (:domain blocks-2-2-blue-red-a1)
(:objects
	a - blue
	b - red

	(:private
		a1 - blueAgent
	)
)
(:init
	(handempty a1)
	(canlift a1 a)
	(clear b)
	(ontable a)
	(on b a)
)
(:goal
	(and
		(on a b)
	)
)
)