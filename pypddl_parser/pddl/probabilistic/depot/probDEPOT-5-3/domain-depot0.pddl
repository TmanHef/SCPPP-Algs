(define (domain depot-5-3-depot0)
	(:requirements :factored-privacy :probabilistic-effects :typing)
    (:types
         place locatable driver - object
         depot distributor - place
         truck hoist surface - locatable
         pallet crate - surface
     )
    (:predicates
        (at ?x - locatable ?y - place)
        (on ?x - crate ?y - surface)
        (in ?x - crate ?y - truck)
        (clear ?x - surface)

        (:private
            (lifting ?agent - place ?x - hoist ?y - crate)
            (available ?agent - place ?x - hoist)
        )
    )

    (:action lift
        :parameters (?p - place ?x - hoist ?y - crate ?z - surface)
        :precondition (and
            (at ?x ?p)
            (available ?p ?x)
            (at ?y ?p)
            (on ?y ?z)
            (clear ?y)
        )
        :effect (probabilistic
        0.8     (and
                    (lifting ?p ?x ?y)
                    (clear ?z)
                    (not (at ?y ?p))
                    (not (clear ?y))
                    (not (available ?p ?x))
                    (not (on ?y ?z))
                ))
        :cost 1
    )


    (:action drop
        :parameters (?p - place ?x - hoist ?y - crate ?z - surface)
        :precondition (and
            (at ?x ?p)
            (at ?z ?p)
            (clear ?z)
            (lifting ?p ?x ?y)
        )
        :effect (and
            (available ?p ?x)
            (at ?y ?p)
            (clear ?y)
            (on ?y ?z)
            (not (lifting ?p ?x ?y))
            (not (clear ?z))
        )
        :cost 1
    )


    (:action load
        :parameters (?p - place ?x - hoist ?y - crate ?z - truck)
        :precondition (and
            (at ?x ?p)
            (at ?z ?p)
            (lifting ?p ?x ?y)
        )
        :effect (probabilistic
         0.8    (and
                    (in ?y ?z)
                    (available ?p ?x)
                    (not (lifting ?p ?x ?y))
                ))
         :cost 1
    )


    (:action unload
        :parameters (?p - place ?x - hoist ?y - crate ?z - truck)
        :precondition (and
            (at ?x ?p)
            (at ?z ?p)
            (available ?p ?x)
            (in ?y ?z)
        )
        :effect (and
            (lifting ?p ?x ?y)
            (not (in ?y ?z))
            (not (available ?p ?x))
        )
        :cost 1
    )

)