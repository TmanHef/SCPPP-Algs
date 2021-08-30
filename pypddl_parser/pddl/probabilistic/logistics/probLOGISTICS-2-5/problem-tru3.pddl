(define (problem logistics-2-5-tru3) (:domain logistics-2-5-tru3)
    (:objects
        p1 - package
        p2 - package
        apt3 - airport
        apt2 - airport
        apt1 - airport
        pos1 - location
        pos2 - location
        pos3 - location

        (:private
            cit3 - city
            tru3 - truck
        )
    )
    (:init
        (vehicle-at tru3 pos3)
        (in-city tru3 pos3 cit3)
        (in-city tru3 apt3 cit3)
        (package-at p1 pos1)
        (package-at p2 pos1)
    )
    (:goal
        (and
            (package-at p1 pos2)
            (package-at p2 pos3)
        )
    )
)