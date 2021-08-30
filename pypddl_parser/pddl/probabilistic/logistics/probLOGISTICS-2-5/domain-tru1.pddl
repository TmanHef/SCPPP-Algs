(define (domain logistics-2-5-tru1)
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
            (in-city ?agent - truck ?loc - location ?city - city)
            (vehicle-at ?veh - vehicle ?loc - location)
        )
    )

    (:action load-truck
        :parameters (?truck - truck ?obj - package ?loc - location)
        :precondition (and
            (vehicle-at ?truck ?loc)
            (package-at ?obj ?loc)
        )
        :effect (probabilistic
            0.9 (and (not (package-at ?obj ?loc)) (in ?obj ?truck))
        )
        :cost 1
    )


    (:action unload-truck
        :parameters (?truck - truck ?obj - package ?loc - location)
        :precondition (and
            (vehicle-at ?truck ?loc)
            (in ?obj ?truck)
        )
        :effect (probabilistic
            0.9 (and (not (in ?obj ?truck)) (package-at ?obj ?loc))
        )
        :cost 1
    )


    (:action drive-truck
        :parameters (?truck - truck ?loc-from - location ?loc-to - location ?city - city)
        :precondition (and
            (vehicle-at ?truck ?loc-from)
            (in-city ?truck ?loc-from ?city)
            (in-city ?truck ?loc-to ?city)
        )
        :effect (probabilistic
            0.8 (and (not (vehicle-at ?truck ?loc-from)) (vehicle-at ?truck ?loc-to))
        )
        :cost 1
    )
)