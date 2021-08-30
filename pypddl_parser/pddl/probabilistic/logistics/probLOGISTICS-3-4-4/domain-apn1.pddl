(define (domain logistics-3-4-4-apn1)
	(:requirements :factored-privacy :probabilistic-effects :typing)
    (:types
	 location vehicle package city - object 
 	 airport - location 
 	 truck airplane - vehicle 
    )

    (:predicates
        (package-at ?obj - package ?loc - location)
        (in ?obj1 - package ?veh - vehicle)

        (:private
            (vehicle-at ?veh - vehicle ?loc - location)
        )
    )

    (:action load-airplane
        :parameters (?airplane - airplane ?obj - package ?loc - airport)
        :precondition (and
            (package-at ?obj ?loc)
            (vehicle-at ?airplane ?loc)
        )
        :effect (probabilistic
			0.9 (and (not (package-at ?obj ?loc)) (in ?obj ?airplane))
		)
		:cost 1
    )


    (:action unload-airplane
        :parameters (?airplane - airplane ?obj - package ?loc - airport)
        :precondition (and
            (in ?obj ?airplane)
            (vehicle-at ?airplane ?loc)
        )
        :effect (probabilistic
            0.9 (and (not (in ?obj ?airplane)) (package-at ?obj ?loc))
        )
        :cost 1
    )


    (:action fly-airplane
        :parameters (?airplane - airplane ?loc-from - airport ?loc-to - airport)
        :precondition
            (vehicle-at ?airplane ?loc-from)
        :effect (probabilistic
            0.8 (and (not (vehicle-at ?airplane ?loc-from)) (vehicle-at ?airplane ?loc-to))
        )
        :cost 1
    )
)