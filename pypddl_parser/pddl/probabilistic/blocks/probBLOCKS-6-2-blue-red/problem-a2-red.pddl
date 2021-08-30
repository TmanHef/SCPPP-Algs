(define (problem BLOCKS-6-2-blue-red-a2) (:domain blocks-6-2-blue-red-a2)
(:objects
	a - blue
	b - red
	c - blue
	d - red
	e - blue
	f - red

	(:private
		a2 - redAgent
	)
)
(:init
	(handempty a2)
	(canlift a2 b)
	(canlift a2 d)
	(canlift a2 f)
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