(define (domain blocks-4-3-blue-red-a3)
	(:requirements :factored-privacy :probabilistic-effects :typing)
	(:types
		agent block - object
		blueAgent redAgent greenAgent - agent
		blue red green - block
	)
	(:predicates
		(on ?x - block ?y - block)
		(ontable ?x - block)
		(clear ?x - block)

		(:private
			(holding ?agent - agent ?x - block)
			(holding-a-stack ?agent - agent ?x - block ?y - block)
			(handempty ?agent - agent)
			(canlift ?a - agent ?x - block)
		)
	)

	; with 3/4 probability - the action succeeds
	; with 1/4 probability - nothing happend
	(:action pick-up
		:parameters (?a - agent ?x - block)
		:precondition (and
			(clear ?x)
			(ontable ?x)
			(handempty ?a)
			(canlift ?a ?x)
		)
		:effect
		(probabilistic
			0.9 (and (not (ontable ?x)) (not (clear ?x)) (not (handempty ?a)) (holding ?a ?x))
		)
		:cost 1
	)

	; with 1/2 probability - the action succeeds
	; with 1/4 probability - the blocks fell onto the table and are no longer stacked
	; with 1/4 probability - nothing happend
	(:action pick-up-a-stack
		:parameters (?a - agent ?x - block ?y - block)
		:precondition (and
			(on ?y ?x)
			(clear ?y)
			(ontable ?x)
			(handempty ?a)
			(canlift ?a ?x)
			(canlift ?a ?y)
		)
		:effect
            (probabilistic
                0.7 (and (not (ontable ?x)) (not (clear ?y)) (not (handempty ?a)) (holding-a-stack ?a ?x ?y))
                0.2 (and (ontable ?x) (ontable ?y) (clear ?x) (not (on ?y ?x)) (handempty ?a) (not (holding-a-stack ?a ?x ?y)))
            )
        :cost 1
	)

	; the action always succeeds
	(:action put-down
		:parameters (?a - agent ?x - block)
		:precondition
			(holding ?a ?x)
		:effect
		(and (not (holding ?a ?x)) (clear ?x) (handempty ?a) (ontable ?x))
		:cost 1
	)

	; with 2/3 probability - the action succeeds
	; with 1/3 probability - the blocks fell onto the table and are no longer stacked
	(:action put-down-a-stack
		:parameters (?a - agent ?x - block ?y - block)
		:precondition
			(and (holding-a-stack ?a ?x ?y) (on ?y ?x))
		:effect
            (probabilistic
                0.7 (and (not (holding-a-stack ?a ?x ?y)) (clear ?y) (handempty ?a) (ontable ?x))
                0.3 (and (not (holding-a-stack ?a ?x ?y)) (ontable ?x) (ontable ?y) (clear ?x) (clear ?y) (not (on ?y ?x)) (handempty ?a))
            )
        :cost 1
	)

	; with 2/3 probability - the action succeeds
	; with 1/3 probability - the agent droped the block on the table
	(:action stack
		:parameters (?a - agent ?x - block ?y - block)
		:precondition (and
			(holding ?a ?x)
			(clear ?y)
		)
		:effect
            (probabilistic
                0.8 (and (not (holding ?a ?x)) (not (clear ?y)) (clear ?x) (handempty ?a) (on ?x ?y))
                0.2 (and (not (holding ?a ?x)) (clear ?x) (clear ?y) (ontable ?x) (handempty ?a))
            )
        :cost 1
	)

	; with 1/3 probability - the action succeeds
	; with 2/3 probability - the agent droped the blocks on the table
	(:action stack-a-stack
		:parameters (?a - agent ?x - block ?y - block ?z - block)
		:precondition (and
			(holding-a-stack ?a ?x ?y)
			(on ?y ?x)
			(clear ?z)
		)
		:effect
            (probabilistic
                0.6 (and (not (holding-a-stack ?a ?x ?y)) (not (clear ?z)) (clear ?y) (handempty ?a) (on ?x ?z))
                0.4 (and (not (holding-a-stack ?a ?x ?y)) (clear ?x) (clear ?y) (ontable ?x) (ontable ?y) (handempty ?a) (not (on ?y ?x)))
            )
        :cost 1
	)

	; with 3/4 probability - the action succeeds
	; with 1/4 probability - nothing happend
	(:action unstack
		:parameters (?a - agent ?x - block ?y - block)
		:precondition (and
			(on ?x ?y)
			(clear ?x)
			(handempty ?a)
			(canlift ?a ?x)
		)
		:effect
		(probabilistic
			0.9 (and (holding ?a ?x) (clear ?y) (not (clear ?x)) (not (handempty ?a)) (not (on ?x ?y)))
		)
		:cost 1
	)

	; with 1/2 probability - the action succeeds
	; with 1/2 probability - the agent droped the blocks on the table
	(:action unstack-a-stack
		:parameters (?a - agent ?x - block ?y - block ?z - block)
		:precondition (and
			(on ?x ?z)
			(on ?y ?x)
			(clear ?y)
			(handempty ?a)
			(canlift ?a ?x)
			(canlift ?a ?y)
		)
		:effect
            (probabilistic
                0.6 (and (holding-a-stack ?a ?x ?y) (clear ?z) (not (clear ?y)) (not (handempty ?a)) (not (on ?x ?z)))
                0.4 (and (not (holding-a-stack ?a ?x ?y)) (clear ?x) (clear ?y) (clear ?z) (not (on ?x ?z)) (not (on ?y ?x)) (ontable ?x) (ontable ?y))
            )
        :cost 1
	)
)