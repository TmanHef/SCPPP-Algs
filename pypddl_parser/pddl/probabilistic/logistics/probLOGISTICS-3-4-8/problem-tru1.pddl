;; best solution length 22 ;;
(define (problem logistics-3-4-8-tru1) (:domain logistics-3-4-8-tru1)
    (:objects
        p1 - package
        p2 - package
        p3 - package
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
        (vehicle-at tru1 pos1)
        (in-city tru1 pos1 cit1)
        (in-city tru1 apt1 cit1)
        (package-at p1 pos1)
        (package-at p2 pos2)
        (package-at p3 pos2)
    )
    (:goal
        (and
            (package-at p1 pos2)
            (package-at p2 pos1)
            (package-at p3 apt1)
        )
    )
)