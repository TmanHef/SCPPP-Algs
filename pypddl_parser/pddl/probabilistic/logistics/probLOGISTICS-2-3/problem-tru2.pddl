(define (problem logistics-2-3-tru2) (:domain logistics-2-3-tru2)
    (:objects
        p1 - package
        p2 - package
        apt2 - airport
        apt1 - airport
        pos1 - location
        pos2 - location

        (:private
            cit2 - city
            tru2 - truck
        )
    )
    (:init
        (package-at p1 pos1)
        (package-at p2 pos2)
        (vehicle-at tru2 pos2)
        (in-city tru2 pos2 cit2)
        (in-city tru2 apt2 cit2)
    )
    (:goal
        (and
            (package-at p1 pos2)
            (package-at p2 pos1)
        )
    )
)