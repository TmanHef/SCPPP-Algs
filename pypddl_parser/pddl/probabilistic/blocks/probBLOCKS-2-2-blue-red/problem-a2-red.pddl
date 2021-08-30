(define (problem BLOCKS-2-2-blue-red-a2) (:domain blocks-2-2-blue-red-a2)
(:objects
	a - blue
	b - red

	(:private
		a2 - redAgent
	)
)
(:init
	(handempty a2)
	(canlift a2 b)
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