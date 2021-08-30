(define (problem logistics-1-3-tru1) (:domain logistics-1-3-tru1)
    (:objects
        p1 - package
        apt2 - airport
        apt1 - airport
        pos1 - location
        pos2 - location

        (:private
            cit1 - city
            tru1 - truck
        )
    )
    (:init
        (package-at p1 pos1)
        (vehicle-at tru1 pos1)
        (in-city tru1 pos1 cit1)
        (in-city tru1 apt1 cit1)
    )
    (:goal
        (and
            (package-at p1 pos2)
        )
    )
)