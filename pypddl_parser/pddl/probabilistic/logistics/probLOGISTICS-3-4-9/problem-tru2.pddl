;; best solution length 26 ;;
(define (problem logistics-3-4-9-tru2) (:domain logistics-3-4-9-tru2)
    (:objects
        p1 - package
        p2 - package
        p3 - package
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
        (vehicle-at tru2 pos2)
        (in-city tru2 pos2 cit2)
        (in-city tru2 apt2 cit2)
        (package-at p1 pos1)
        (package-at p2 pos2)
        (package-at p3 pos2)
    )
    (:goal
        (and
            (package-at p1 pos2)
            (package-at p2 pos1)
            (package-at p3 pos1)
        )
    )
)