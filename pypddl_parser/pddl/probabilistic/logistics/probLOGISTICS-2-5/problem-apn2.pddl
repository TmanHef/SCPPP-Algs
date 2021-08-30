(define (problem logistics-2-5-apn2) (:domain logistics-2-5-apn2)
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
            apn2 - airplane
        )
    )
    (:init
        (vehicle-at apn2 apt2)
        (package-at p1 pos1)
        (package-at p2 pos1)
    )
    (:goal
        (and
            (package-at p1 pos2)
            (package-at p2 pos1)
        )
    )
)